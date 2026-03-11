using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Tuanjie.NeuralDeformer
{
    /// <summary>
    /// NeuralDeformerPlayer class manages the deformation and neural network inference on the mesh
    /// using Sentis on both CPU and GPU.
    ///  Deformer pipeline:
    ///  Neutral Mesh -> Sentis Inference -> Apply Deformation -> Skinning -> Recalculate Normals -> Rendering
    /// </summary>
    [ExecuteAlways, RequireComponent(typeof(SkinnedMeshRenderer))]
    [AddComponentMenu("Neural Deformer/Neural Deformer Player")]
    public class NeuralDeformerPlayer : MeshInstanceBehaviour
    {
#if UNITY_EDITOR
        /// <summary>
        /// List of all enabled NeuralDeformerPlayer instances in the editor.
        /// </summary>
        public static List<NeuralDeformerPlayer> enabledInstances =
            new List<NeuralDeformerPlayer>();
        public static readonly object enabledInstancesLock = new object();
#endif
        // Private fields for managing disposable objects and GPU resources
        private List<FieldInfo> _disposableFields;
        private List<FieldInfo> _listDisposableFields;
        private List<FieldInfo> _gpuResourceDisposableFields;
        private List<FieldInfo> _gpuInferenceResourceDisposableFields;
        private List<FieldInfo> _gpuPostprocessingResourceDisposableFields;

        /// <summary>
        /// Event that is invoked after the deformation process is completed.
        /// </summary>
        public event Action onDeform;

        #region Deformation Inspector Configurations
        /// <summary>
        /// Toggle to enable or disable the mesh deformation.
        /// </summary>
        [Header("Enable/Disable Deformation")]
        public bool enableDeformation = false;

        /// <summary>
        /// (Required) Metadata information for the deformer data.
        /// </summary>
        [Header("Deformation Setup")]
        [SerializeField]
        public NeuralDeformerDatasetMetaInfo neuralDeformerDatasetMetaInfo;

        /// <summary>
        /// (Required) Model asset (*.onnx or *.sentis) for the deformation inference.
        /// </summary>
        [SerializeField]
        public TextAsset modelAsset;

        /// <summary>
        /// Backend type used for Sentis model inference (CPU or GPU).
        /// </summary>
        [Header("Deformation Settings")]
        [SerializeField]
        private BackendType _inferenceBackend = BackendType.GPU;

        /// <summary>
        /// Backend type used for deform (apply deformation on the mesh & recalculate normals).
        /// </summary>
        [SerializeField]
        private BackendType _deformBackend = BackendType.GPU;

        /// <summary>
        /// Deformation weight that affects the intensity of the deformation.
        /// </summary>
        [SerializeField]
        [Range(0.0f, 1.0f)]
        public float deformationWeight = 0.95f;

        /// <summary>
        /// Influence of the alpha mask on the deformation weight.
        /// </summary>
        [SerializeField]
        [Range(0.0f, 1.0f)]
        public float alphaMaskInfluenceWeight = 0.55f;

        /// <summary>
        /// Flag to determine whether to recalculate normals during deformation.
        /// </summary>
        [SerializeField]
        public bool recalculateNormals = true;

        /// <summary>
        /// Enable or disable debug & profiling for the deformation process.
        /// </summary>
        [Header("Deformation Debug")]
        [SerializeField]
        public bool enableDeformationDebuging = false;

        /// <summary>
        /// Enable or disable debug logging in console for the deformation process.
        /// </summary>
        [SerializeField]
        public bool enableDeformationLogging = false;
        #endregion


        #region Sentis Runtime Data Resources
        // Resources for managing the model and worker during inference.

        /// <summary>
        /// List of transforms for the mesh joints.
        /// Auto loads from <see cref="NeuralDeformerDatasetMetaInfo"/>.
        /// </summary>
        [NonSerialized]
        private List<Transform> _joints;

        /// <summary>
        /// Sentis neural network model instantce, which loads from the model asset.
        /// See: https://docs.unity.cn/Packages/com.unity.sentis@2.1/api/Unity.Sentis.Model.html
        /// </summary>
        [NonSerialized]
        private Model _model;

        /// <summary>
        /// Worker to execute the deformation neural network.
        /// See: https://docs.unity.cn/Packages/com.unity.sentis@2.1/api/Unity.Sentis.Worker.html
        /// </summary>
        [NonSerialized]
        [DisposeOnDestroy]
        private Worker _worker;

        /// <summary>
        // Array of input tensors for the Sentis model.
        /// </summary>
        [NonSerialized]
        [ListDisposeOnDestroy]
        private Tensor[] _inputTensors;

        /// <summary>
        /// List of model inputs.
        /// See: https://docs.unity.cn/Packages/com.unity.sentis@2.1/api/Unity.Sentis.Model.Input.html
        /// </summary>
        [NonSerialized]
        private List<Model.Input> _modelInputs;

        /// <summary>
        /// List of model outputs.
        /// See: https://docs.unity.cn/Packages/com.unity.sentis@2.1/api/Unity.Sentis.Model.Output.html
        /// </summary>
        [NonSerialized]
        private List<Model.Output> _modelOutputs;
        #endregion


        #region Mesh & Topology Data Resources
        // Mesh data and buffers for mesh deformation

        /// <summary>
        /// Mesh buffers for mesh deformation.
        /// </summary>
        [NonSerialized]
        private MeshBuffers _meshBuffers;

        /// <summary>
        /// Mesh adjacency information for mesh deformation.
        /// </summary>
        [NonSerialized]
        private MeshAdjacency _meshAdjacency;

        /// <summary>
        /// Skinned mesh renderer component for mesh rendering.
        /// </summary>
        [NonSerialized]
        private SkinnedMeshRenderer _skinnedMeshRenderer;

        /// <summary>
        /// Unique vertex index mapping.
        /// </summary>
        [NonSerialized, DisposeOnDestroy]
        private NativeArray<int> _vertexMapping;

        /// <summary>
        /// Stride of the color buffer for mesh deformation.
        /// </summary>
        [NonSerialized]
        private int _colorBufferStride;

        /// <summary>
        /// Offset of the color buffer for mesh deformation.
        /// </summary>
        [NonSerialized]
        private int _colorBufferOffset;

        /// <summary>
        /// Color buffer for mesh deformation.
        /// </summary>
        [NonSerialized]
        private uint[] _colorBuffer;

        /// <summary>
        /// Flag indicating if alpha masked deformation weight is enabled.
        /// </summary>
        [NonSerialized]
        private bool _alphaMaskedDeformationWeightEnabled;

        /// <summary>
        /// Vertex deltas for mesh deformation.
        /// </summary>
        [NonSerialized]
        private Vector3[] _vertexDeltas;

        /// <summary>
        /// Deformed vertex positions after mesh deformation.
        /// </summary>
        [NonSerialized]
        private Vector3[] _deformedVertexPositions;

        /// <summary>
        /// CPU mesh buffer for mesh deformation.
        /// </summary>
        [NonSerialized]
        private float[] _cpuMeshBuffer;

        /// <summary>
        /// CPU vertex positions for mesh deformation.
        /// </summary>
        [NonSerialized]
        private Vector3[] _cpuVertexPositions;

        /// <summary>
        /// CPU vertex normals for mesh deformation.
        /// </summary>
        [NonSerialized]
        private List<Vector3> _cpuVertexNormals;

        /// <summary>
        /// Baked mesh for mesh deformation.
        /// </summary>
        [NonSerialized]
        private Mesh _bakedMesh;

        /// <summary>
        /// Stream index for the buffer used in mesh deformation.
        /// </summary>
        [NonSerialized]
        private int _bufferStream;

        /// <summary>
        /// Byte stride of the buffer used in mesh deformation.
        /// </summary>
        [NonSerialized]
        private int _byteBufferStride;

        /// <summary>
        /// Float stride of the buffer used in mesh deformation.
        /// </summary>
        [NonSerialized]
        private int _floatBufferStride;

        /// <summary>
        /// Byte offset for vertex positions in the buffer.
        /// </summary>
        [NonSerialized]
        private int _bytePositionOffset;

        /// <summary>
        /// Byte offset for vertex normals in the buffer.
        /// </summary>
        [NonSerialized]
        private int _byteNormalOffset;

        /// <summary>
        /// Float offset for vertex positions in the buffer.
        /// </summary>
        [NonSerialized]
        private int _floatPositionOffset;

        /// <summary>
        /// Float offset for vertex normals in the buffer.
        /// </summary>
        [NonSerialized]
        private int _floatNormalOffset;

        #endregion


        #region GPU Compute Data Resources
        // GPU resources and compute shaders for deformation

        private static string s_computeShaderName = "DeformCS";

        private static readonly object s_computeShaderLock = new object();

        /// <summary>
        /// Compute shader for deformation operations.
        /// </summary>
        [DisposeOnDestroy]
        private static ComputeShader s_computeShader;

        /// <summary>
        /// Graphics buffer for GPU color data.
        /// </summary>
        [NonSerialized]
        [GPUResourceDisposeOnDestroy]
        private GraphicsBuffer _gpuColorBuffer;

        /// <summary>
        /// Graphics buffer for vertex positions used in GPU inference.
        /// </summary>
        [NonSerialized]
        [GPUInferenceResourceDisposeOnDestroy]
        private GraphicsBuffer _vertexPositionsBuffer;

        /// <summary>
        /// Graphics buffer for unique vertex index mapping.
        /// </summary>
        [NonSerialized]
        [GPUInferenceResourceDisposeOnDestroy]
        private GraphicsBuffer _vertexMappingBuffer;

#if UNITY_EDITOR
        /// <summary>
        /// Graphics buffer for temporary mesh data used in CPU skinning (editor mode only).
        /// </summary>
        [NonSerialized]
        [GPUInferenceResourceDisposeOnDestroy]
        private GraphicsBuffer _gpuTempMeshBuffer;
#endif

        /// <summary>
        /// Graphics buffer for triangle normals used in post-processing.
        /// </summary>
        [NonSerialized]
        [GPUPostprocessingResourceDisposeOnDestroyAttribute]
        private GraphicsBuffer _triangleNormalBuffer;

        /// <summary>
        /// Graphics buffer for mesh triangle indices used in post-processing.
        /// </summary>
        [NonSerialized]
        [GPUPostprocessingResourceDisposeOnDestroyAttribute]
        private GraphicsBuffer _meshTriangleIndexBuffer;

        /// <summary>
        /// Graphics buffer for mesh adjacency triangle indices used in post-processing.
        /// </summary>
        [NonSerialized]
        [GPUPostprocessingResourceDisposeOnDestroyAttribute]
        private GraphicsBuffer _meshAdjacencyTriangleIndexBuffer;

        /// <summary>
        /// Graphics buffer for mesh adjacency triangle index offsets used in post-processing.
        /// </summary>
        [NonSerialized]
        [GPUPostprocessingResourceDisposeOnDestroyAttribute]
        private GraphicsBuffer _meshAdjacencyTriangleIndexOffsetBuffer;

        /// <summary>
        /// Graphics buffer for mesh adjacency triangle index strides used in post-processing.
        /// </summary>
        [NonSerialized]
        [GPUPostprocessingResourceDisposeOnDestroyAttribute]
        private GraphicsBuffer _meshAdjacencyTriangleIndexStrideBuffer;
        #endregion


        #region Runtime States
        // States for managing deformation and backend operations

        [NonSerialized]
        private BackendType _sentisInferenceBackend;

        [NonSerialized]
        private BackendType _deformComputeBackend;

        [NonSerialized]
        private bool _initialized = false;

        [NonSerialized]
        private bool _deformationApplied = false;

        private bool _isInferenceBackendChanged =>
            _sentisInferenceBackend != _inferenceBackend && enableDeformation;

        private bool _isDeformBackendChanged =>
            _deformComputeBackend != _deformBackend && enableDeformation;

        private bool _isGPUEnabled =>
            (_sentisInferenceBackend == BackendType.GPU || _deformComputeBackend == BackendType.GPU)
            && enableDeformation;

        private bool _isDeformationAvailable => CheckDeformationAvailable();

        private bool _enableDeformationDebugLogging =>
            enableDeformationDebuging && enableDeformationLogging;
        #endregion


        #region Life Cycle
        /// <summary>
        /// Called when the component is enabled. Initializes necessary resources and applies deformation if available.
        /// </summary>
        void OnEnable()
        {
            InitializeAllDisposableFields();
            LoadSkinnedMeshRenderer();

#if UNITY_EDITOR
            lock (enabledInstancesLock)
            {
                if (NeuralDeformerPlayer.enabledInstances.Contains(this) == false)
                    NeuralDeformerPlayer.enabledInstances.Add(this);
            }
#endif
            if (enableDeformation && !_isDeformationAvailable)
            {
                enableDeformation = false;
            }
            else
            {
                LogPerformance(InitDataResources);
            }

            RenderPipelineManager.beginContextRendering -= AfterSkinningCallback;
            RenderPipelineManager.beginContextRendering += AfterSkinningCallback;
        }

        /// <summary>
        /// This method is called every frame to perform inference and apply deformation to the mesh.
        /// It also manages the resources for both GPU and CPU processing,
        /// which checks if deformation is enabled and if the necessary resources are available. 
        /// If so, it applies deformation using either <see cref="DeformGPU"/> or <see cref="DeformCPU"/>, 
        /// depending on the backend type <see cref="_sentisInferenceBackend"/> set：These 2 methods perform
        /// deformation by utilizing the Sentis neural network to infer deformation deltas,
        /// which are then applied to the neutral pose using either <see cref="ApplyDeformCPU"/>
        /// or <see cref="ApplyDeformGPU"/>, depending on the value of <see cref="_deformComputeBackend"/>.
        /// </summary>
        void LateUpdate()
        {
            if (enableDeformation && !_isDeformationAvailable)
            {
                enableDeformation = false;
                return;
            }

            if (_isInferenceBackendChanged || _isDeformBackendChanged || !_initialized)
            {
                LogPerformance(ResetDataResources);
            }

            if (enableDeformation)
            {
                if (_sentisInferenceBackend == BackendType.GPU)
                {
                    // Normal recalculation in GPU mode is deferred to the AfterSkinningCallback:
                    // This is necessary because we need to wait for GPU skinning to complete before correctly calculating normals on the deformed mesh
                    // The RecalculateNormalsGPU() method will be called when the RenderPipelineManager.beginContextRendering event is triggered
                    // This ensures that normal calculation occurs at the proper stage in the rendering pipeline, avoiding conflicts with the GPU skinning process
                    // Thus, only DeformGPU() is invoked here
                    LogPerformance(DeformGPU);
                }
                else
                {
                    // Normal recalculation in CPU mode is also deferred to the AfterSkinningCallback
                    // Only DeformCPU() is invoked here
                    LogPerformance(DeformCPU);
                }
                _deformationApplied = true;
            }
            else
            {
                if (_deformationApplied)
                {
                    // Refresh mesh instance
                    RemoveMeshInstance();
                    EnsureMeshInstance();
                    LoadMeshAsset();

                    // Reset the mesh to its original state
                    NoDeformCPU();
                    _deformationApplied = false;
                }
            }
        }

        /// <summary>
        /// Called when the component is disabled. Frees all resources.
        /// </summary>
        private void OnDisable()
        {
            RenderPipelineManager.beginContextRendering -= AfterSkinningCallback;
#if UNITY_EDITOR
            lock (enabledInstancesLock)
            {
                if (NeuralDeformerPlayer.enabledInstances.Contains(this))
                    NeuralDeformerPlayer.enabledInstances.Remove(this);
            }
#endif
            FreeDataResources();
        }

        /// <summary>
        /// Callback function that is invoked after skinning is completed.
        /// This method is registered with RenderPipelineManager.beginContextRendering and is called
        /// at the beginning of each render context.
        /// It is used to recalculate normals after skinning and deformation have been applied,
        /// ensuring that the mesh normals are correctly updated to reflect the deformed geometry.
        /// Therefore, to recalculate normals, this method calls either <see cref="RecalculateNormalsCPU"/>
        /// or <see cref="RecalculateNormalsGPU"/>, depending on the value of <see cref="_deformComputeBackend"/>.
        /// </summary>
        /// <param name="scriptableRenderContext">The scriptable render context for the current frame.</param>
        /// <param name="cameras">The list of cameras being rendered in the current context.</param>
        void AfterSkinningCallback(
            ScriptableRenderContext scriptableRenderContext,
            List<Camera> cameras
        )
        {
            // Check if deformation is enabled
            if (enableDeformation)
            {
                // Determine the backend type for recalculating normals
                if (_deformComputeBackend == BackendType.GPU)
                {
                    LogPerformance(RecalculateNormalsGPU);
                }
                else
                {
                    LogPerformance(RecalculateNormalsCPU);
                }
            }
            // Invoke the onDeform event if it has been subscribed to
            onDeform?.Invoke();
        }
        #endregion


        #region Data Resources Management

        /// <summary>
        /// Notify this component to reset its data resources in next frame.
        /// </summary>
        public void NotifyResetDataResources()
        {
            if (_initialized)
                _initialized = false;
        }

        /// <summary>
        /// Resets all data resources by freeing the old ones and initializing new ones.
        /// </summary>
        private void ResetDataResources()
        {
            DebugLog("DeformerSentis Reset Resources.");
            FreeDataResources();
            InitDataResources();
        }

        /// <summary>
        /// Initializes all necessary data resources for mesh deformation. This includes loading the compute shader,
        /// configuring the Sentis model and its resources, ensuring the mesh instance is available, and loading the mesh asset.
        /// It also loads GPU data resources if the GPU backend is enabled.
        /// </summary>
        private void InitDataResources()
        {
            DebugLog("NeuralDeformerPlayer.InitDataResources()");

            // Apply the backend change (switch between CPU and GPU).
            ApplyBackendChange();

            // Load the compute shader to be used for GPU-based deformation.
            LoadComputeShader();

            // Load the Sentis model, inputs, and outputs for deformation.
            LoadSentisDataResources();

            // Ensure that the mesh instance is available and ready for modification.
            EnsureMeshInstance();

            // Load the mesh asset (geometry, vertices, etc.).
            LoadMeshAsset();

            // If the GPU backend is enabled, load additional GPU resources for deformation.
            if (_isGPUEnabled)
            {
                LoadGPUDataResources();
            }

            // Mark the data resources as initialized.
            _initialized = true;
        }

        /// <summary>
        /// Called when the mesh instance is created.
        /// This method retrieves the vertex attribute streams for position, normal, and tangent,
        /// and ensures that they are stored in the same buffer.
        /// It then calculates the buffer stride and attribute offsets for position and normal.
        /// This function is invoked by the base class method <see cref="EnsureMeshInstance"/>,
        /// which ensures that the mesh instance is available and ready for modification.
        /// </summary>
        internal protected override void OnMeshInstanceCreated()
        {
            // Get the vertex attribute streams for position, normal, and tangent.
            int positionStream = meshInstance.GetVertexAttributeStream(VertexAttribute.Position);
            int normalStream = meshInstance.GetVertexAttributeStream(VertexAttribute.Normal);
            int tangentStream = meshInstance.GetVertexAttributeStream(VertexAttribute.Tangent);
            // Ensure that position, normal, and tangent are in the same buffer.
            Assert.IsTrue(
                positionStream == normalStream && positionStream == tangentStream,
                $"Invalid vertex attribute streams: "
                    + $"positions = {positionStream}, normals = {normalStream} "
                    + $"and tangents = {tangentStream} must be in same buffer"
            );

            _bufferStream = positionStream;

            _byteBufferStride = meshInstance.GetVertexBufferStride(_bufferStream);
            _floatBufferStride = _byteBufferStride / sizeof(float);

            _bytePositionOffset = meshInstance.GetVertexAttributeOffset(VertexAttribute.Position);
            _floatPositionOffset = _bytePositionOffset / sizeof(float);

            _byteNormalOffset = meshInstance.GetVertexAttributeOffset(VertexAttribute.Normal);
            _floatNormalOffset = _byteNormalOffset / sizeof(float);
        }

        /// <summary>
        /// Frees all allocated data resources and sets the component's state to uninitialized.
        /// This function removes the mesh instance, disposes of disposable objects, and resets the initialization flag.
        /// It ensures that any dynamically allocated resources are properly released when the deformation is no longer required.
        /// </summary>
        private void FreeDataResources()
        {
            DebugLog("NeuralDeformerPlayer.FreeDataResources()");

            // Remove the mesh instance from the current object
            RemoveMeshInstance();

            // Free all disposable objects to release memory and GPU resources
            FreeAllDisposableObjects();

            // Set the _initialized flag to false to indicate that the resources are not initialized
            _initialized = false;
        }

        /// <summary>
        /// Applies the backend change if deformation is enabled. This method ensures that the deformation process 
        /// switches to the specified backend (CPU or GPU) when deformation is enabled.
        /// </summary>
        private void ApplyBackendChange()
        {
            if (enableDeformation)
            {
                DebugLog(
                    $"NeuralDeformerPlayer.ApplyBackendChange(): _sentisInferenceBackend({_sentisInferenceBackend} -> {_inferenceBackend})"
                );
                DebugLog(
                    $"NeuralDeformerPlayer.ApplyBackendChange(): _deformComputeBackend({_deformComputeBackend} -> {_deformBackend})"
                );
                // Switch the backend to the newly selected backend (CPU or GPU)
                _sentisInferenceBackend = _inferenceBackend;
                _deformComputeBackend = _deformBackend;
            }
        }

        /// <summary>
        /// Loads the compute shader from the Resources folder if not already loaded.
        /// This function is responsible for caching the compute shader for deformation operations.
        /// It also initializes the shader properties for the various uniform and kernel fields.
        /// </summary>
        private static void LoadComputeShader()
        {
            // If the compute shader is already loaded, no need to load it again
            if (s_computeShader != null)
            {
                return;
            }

            // Lock to ensure thread safety while loading the compute shader
            lock (s_computeShaderLock)
            {
                // Load the compute shader from the Resources folder
                s_computeShader = Resources.Load<ComputeShader>(s_computeShaderName);

                // Initialize the shader uniform properties with their corresponding shader property IDs
                foreach (var field in typeof(Uniforms).GetFields())
                {
                    field.SetValue(null, Shader.PropertyToID(field.Name));
                }

                // Initialize the kernel properties with the corresponding kernel IDs
                foreach (var field in typeof(Kernels).GetFields())
                {
                    field.SetValue(null, s_computeShader.FindKernel(field.Name));
                }
            }
        }

        /// <summary>
        /// Initializes and loads GPU-specific data resources required for mesh deformation.
        /// This method prepares the GPU buffers and ensures that necessary resources for deformation,
        /// such as vertex positions and unique vertex index mappings, are available for GPU processing.
        /// </summary>
        private void LoadGPUDataResources()
        {
            DebugLog("NeuralDeformerPlayer.LoadGPUDataResources()");

            // Enable raw buffer for reading/writing by GPU if necessary
            meshInstance.vertexBufferTarget |= GraphicsBuffer.Target.Raw;

            // Check if the necessary buffers for GPU inference are already loaded, if not, load them
            if (_vertexPositionsBuffer == null || _vertexMappingBuffer == null)
                LoadGPUInferenceBufferResources();

            // If the post-processing buffers are not loaded yet, load them
            if (
                _triangleNormalBuffer == null
                || _meshTriangleIndexBuffer == null
                || _meshAdjacencyTriangleIndexBuffer == null
                || _meshAdjacencyTriangleIndexOffsetBuffer == null
                || _meshAdjacencyTriangleIndexStrideBuffer == null
            )
                LoadGPUPostprocessingBufferResources();
        }

        /// <summary>
        /// Loads the mesh asset data into the deformer instance.
        /// This method ensures that the mesh buffers are properly initialized or reloaded if necessary.
        /// It verifies that the mesh buffers, especially the vertex positions, are properly assigned and initialized.
        /// </summary>
        private void LoadMeshAsset()
        {
            // If mesh buffers have not been initialized, initialize them
            if (_meshBuffers == null)
            {
                _meshBuffers = new MeshBuffers(meshAsset);
            }
            else
            {
                // If they are already initialized, reload the data from the asset
                _meshBuffers.LoadFrom(meshAsset);
            }

            // Ensure that the mesh buffers and vertex positions are properly initialized
            Assert.IsTrue(
                _meshBuffers != null && _meshBuffers.vertexPositions != null,
                "Mesh buffers or vertex positions are not initialized."
            );
        }

        /// <summary>
        /// Loads and initializes the <see cref="SkinnedMeshRenderer"/> component if it's not already initialized.
        /// Ensures that the component is attached to the GameObject and the shared mesh is valid.
        /// </summary>
        private void LoadSkinnedMeshRenderer()
        {
            if (
                _skinnedMeshRenderer == null
                || _skinnedMeshRenderer.sharedMesh == null
                || _skinnedMeshRenderer.sharedMesh.GetInstanceID() >= 0
            )
            {
                _skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
            }
            Assert.IsTrue(
                _skinnedMeshRenderer != null,
                "Resource Not Available: `_skinnedMeshRenderer` not initialized,"
                    + " please check if the `Mesh Renderer` component has added to the game object"
            );
        }

        /// <summary>
        /// Initializes the Sentis model and its associated input/output tensors for the deformation process.
        /// This function loads the model, prepares the input tensors, and sets up the worker for inference.
        /// Here, vertex table <see cref="NeuralDeformerDatasetMetaInfo.GetVertexTable()"/> is loaded and used to efficiently
        /// apply the deformed positions to the mesh vertices.
        /// </summary>
        private void LoadSentisDataResources()
        {
            DebugLog("NeuralDeformerPlayer.LoadSentisDataResources()");

            if (modelAsset == null)
                return;

            // Load the model from the provided model asset
            _model = ModelLoader.Load(modelAsset);
            // Get the inputs required by the model
            _modelInputs = _model.inputs;
            // Get the outputs from the model, sorted by name
            _modelOutputs = _model.outputs.OrderBy(o => o.name).ToList();
            // Initialize the input tensors array
            _inputTensors = new Tensor[_modelInputs.Count];

            // Set up the worker based on the selected backend (CPU or GPU)
            _worker = _sentisInferenceBackend switch
            {
                BackendType.CPU => new Worker(_model, Unity.Sentis.BackendType.CPU),
                BackendType.GPU => new Worker(_model, Unity.Sentis.BackendType.GPUCompute),
                _ => new Worker(_model, Unity.Sentis.BackendType.CPU),
            };

            // Initialize each input tensor with the appropriate shape
            for (int i = 0; i < _modelInputs.Count; i++)
            {
                var input = _modelInputs[i];
                var shape = ConvertToTensorShape(input.shape);
                _inputTensors[i] = new Tensor<float>(shape);
            }

            if (_enableDeformationDebugLogging)
            {
                var tensorShapeInfos = new List<string>();
                for (int i = 0; i < _inputTensors.Length; i++)
                {
                    tensorShapeInfos.Add($"{string.Join(", ", _inputTensors[i].shape)}");
                }
                Debug.LogFormat(
                    "Model Inputs: {0} Input Tensors with Shape [{1}]",
                    _inputTensors.Length,
                    string.Join(", ", tensorShapeInfos)
                );
                var modelOuputInfos = new List<string>();
                for (int i = 0; i < _modelOutputs.Count; i++)
                {
                    modelOuputInfos.Add($"{_modelOutputs[i].name}, {_modelOutputs[i].index}");
                }
                Debug.LogFormat("Model Outputs: [{0}]", string.Join(", ", modelOuputInfos));
            }

            // Get the unique vertex index mapping from the deformation data meta infos
            _vertexMapping = neuralDeformerDatasetMetaInfo.GetVertexTable();
            Assert.IsTrue(
                _vertexMapping.Length == neuralDeformerDatasetMetaInfo.VertexCount,
                $"Resource Not Available: the length of `_vertexMapping` is expected to be {neuralDeformerDatasetMetaInfo.VertexCount}" +
                    $"but got {_vertexMapping.Length}"
            );

            // Get the joints data from the deformation data meta infos
            var succ = neuralDeformerDatasetMetaInfo.TryGetJointTransformList(ref _joints);
            // Get the joints data from the deformation data meta infos
            Assert.IsTrue(succ, "Resource Not Available: `joints` is empty, failed to load Joints from neuralDeformerDatasetMetaInfo");
        }

        /// <summary>
        /// Loads the color buffer resources from the mesh to handle alpha masked deformation.
        /// If vertex color stream is present, this function extracts the color data from the mesh, 
        /// determines whether alpha mask influence is enabled, and sets up the color buffer accordingly.
        /// </summary>
        private void LoadColorBufferResources()
        {
            // Retrieve the vertex color stream from the mesh
            int vertexColorStream = meshInstance.GetVertexAttributeStream(VertexAttribute.Color);
            if (vertexColorStream == -1 || alphaMaskInfluenceWeight == 0f)
            {
                // If no vertex color stream exists and alpha mask influence weight is set, disable the color buffer
                _colorBufferStride = 0;
                _colorBufferOffset = 0;
                _colorBuffer = default;

                _gpuColorBuffer?.Dispose();
                _gpuColorBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, 1, 4);

                _alphaMaskedDeformationWeightEnabled = false;
            }
            else
            {
                // If vertex color stream is found, read the color buffer data from the mesh
                using (
                    Mesh.MeshDataArray colorBufferArray = Mesh.AcquireReadOnlyMeshData(meshInstance)
                )
                {
                    var vertexData = colorBufferArray[0].GetVertexData<uint>(vertexColorStream);
                    _colorBuffer ??= new uint[vertexData.Length];
                    vertexData.CopyTo(_colorBuffer);
                }
                _gpuColorBuffer?.Dispose();
                _gpuColorBuffer = meshInstance.GetVertexBuffer(vertexColorStream);

                // Retrieve the stride and offset values for the color buffer
                _colorBufferStride = meshInstance.GetVertexBufferStride(vertexColorStream);
                _colorBufferOffset = meshInstance.GetVertexAttributeOffset(VertexAttribute.Color);

                _alphaMaskedDeformationWeightEnabled = true;
            }
        }

        /// <summary>
        /// Loads the GPU inference buffer resources required for the mesh deformation process.
        /// This includes creating and setting up GPU buffers for vertex positions and unique vertex index mappings.
        /// </summary>
        private void LoadGPUInferenceBufferResources()
        {
            DebugLog("NeuralDeformerPlayer.LoadGPUInferenceBufferResources()");

            // Iterate through the GPU inference resource disposable fields and free them
            foreach (var field in _gpuInferenceResourceDisposableFields)
            {
                FreeDisposableObject(field, true);
            }

            // Create a GraphicsBuffer for storing vertex positions, structured as a float3 type (x, y, z)
            _vertexPositionsBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                _meshBuffers.vertexCount,
                UnsafeUtility.SizeOf<float3>()
            );

            // Create a GraphicsBuffer for storing unique vertex index mappings, with each entry being a uint
            _vertexMappingBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                neuralDeformerDatasetMetaInfo.VertexCount,
                UnsafeUtility.SizeOf<uint>()
            );

            // Set the data for the created buffers
            _vertexPositionsBuffer.SetData(_meshBuffers.vertexPositions);
            _vertexMappingBuffer.SetData(_vertexMapping);
        }

        /// <summary>
        /// Loads and initializes GPU postprocessing buffer resources required for the deformation process.
        /// This includes creating and setting data for buffers related to triangle normals and mesh adjacency.
        /// It also ensures that the adjacency-related data, such as triangle indices, are properly set up for GPU processing.
        /// </summary>
        private void LoadGPUPostprocessingBufferResources()
        {
            DebugLog("NeuralDeformerPlayer.LoadGPUPostprocessingBufferResources()");

            // Free any previously allocated GPU postprocessing resources
            foreach (var field in _gpuPostprocessingResourceDisposableFields)
            {
                FreeDisposableObject(field, true);
            }

            // Initialize mesh adjacency for triangle connectivity data
            _meshAdjacency = new MeshAdjacency(_meshBuffers, false);

            // Create GPU buffers for various postprocessing data such as triangle normals and adjacency data
            _triangleNormalBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                _meshAdjacency.triangleCount,
                UnsafeUtility.SizeOf<float3>()
            );
            _meshTriangleIndexBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                _meshAdjacency.triangleCount * 3,
                sizeof(uint)
            );
            _meshAdjacencyTriangleIndexBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                _meshAdjacency.vertexTriangles.itemCount,
                sizeof(uint)
            );
            _meshAdjacencyTriangleIndexOffsetBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                _meshAdjacency.vertexCount,
                sizeof(uint)
            );
            _meshAdjacencyTriangleIndexStrideBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Structured,
                _meshAdjacency.vertexCount,
                sizeof(uint)
            );
#if UNITY_EDITOR
            _gpuTempMeshBuffer = new GraphicsBuffer(
                GraphicsBuffer.Target.Raw,
                _meshAdjacency.vertexCount * _floatBufferStride,
                sizeof(float)
            );
#endif

            // Prepare adjacency data for triangles and mesh vertices
            NativeArray<uint> adjacentTriangleIndices = new NativeArray<uint>(
                _meshAdjacency.vertexTriangles.itemCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory
            );
            NativeArray<uint> meshAdjacencyTriangleOffsets = new NativeArray<uint>(
                _meshAdjacency.vertexCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory
            );
            NativeArray<uint> meshAdjacencyTriangleStrides = new NativeArray<uint>(
                _meshAdjacency.vertexCount,
                Allocator.Temp,
                NativeArrayOptions.UninitializedMemory
            );

            // Populate the adjacency-related data for triangles and vertices
            uint offset = 0;
            for (int i = 0; i < _meshAdjacency.vertexCount; i++)
            {
                uint stride = 0;
                foreach (uint triangle in _meshAdjacency.vertexTriangles[i])
                {
                    int index = (int)(offset + stride);
                    adjacentTriangleIndices[index] = triangle;
                    stride++;
                }
                meshAdjacencyTriangleOffsets[i] = offset;
                meshAdjacencyTriangleStrides[i] = stride;
                offset += stride;
            }

            // Set the data for triangle indices and adjacency buffers
            _meshTriangleIndexBuffer.SetData(_meshBuffers.triangles);
            _meshAdjacencyTriangleIndexBuffer.SetData(adjacentTriangleIndices);
            _meshAdjacencyTriangleIndexOffsetBuffer.SetData(meshAdjacencyTriangleOffsets);
            _meshAdjacencyTriangleIndexStrideBuffer.SetData(meshAdjacencyTriangleStrides);

            // Dispose of temporary arrays after use
            adjacentTriangleIndices.Dispose();
            meshAdjacencyTriangleOffsets.Dispose();
            meshAdjacencyTriangleStrides.Dispose();
        }
        #endregion


        #region CPU Inference & Postprocessing
        /// <summary>
        /// Applies the deformation to the mesh vertices when the backend type is set to CPU.
        /// This function processes the neural network output, which contains vertex deformation delta data,
        /// and applies the deformed vertex positions to the mesh. The deformation is applied based on the output 
        /// of the Sentis model, which computes the changes in vertex positions (deltas) for each unique vertex.
        /// 
        /// The deformation is influenced by the following parameters:
        /// - <see cref="deformationWeight"/>: The weight that affects how strongly the deformation is applied.
        /// - <see cref="alphaMaskInfluenceWeight"/>: The weight that influences the alpha mask's effect on deformation.
        /// - Alpha Masking: If alpha mask influence is enabled, the vertex color's alpha value (from the vertex color buffer) 
        ///   is used to modulate the deformation weight for each vertex, enabling partial deformation based on the mask.
        /// See more in <see cref="ApplyDeformCPU()"/> or <see cref="ApplyDeformGPU(CommandBuffer)"/>.
        /// 
        /// The deformation process follows these steps:
        /// 1. Prepare input tensors based on joint transformations via <see cref="PrepareSentisInferenceInputs()"/>.
        /// 2. Schedule the inference task using the <see cref="_worker"/> on <see cref="Unity.Sentis.BackendType.CPU()"/>.
        /// 3. Depending on the selected backend type in <see cref="_deformBackend"/>, apply either CPU or GPU deformation.
        /// 4. Apply deformation:
        ///     - For CPU, <see cref="ApplyDeformCPU()"/>;
        ///     - for GPU, <see cref="ApplyDeformGPU(CommandBuffer)"/>.
        ///
        /// This function ensures that deformation is computed only when the backend type is <see cref="Unity.Sentis.BackendType.CPU"/>.
        /// </summary>
        void DeformCPU()
        {
            DebugLog("NeuralDeformerPlayer.DeformCPU()");

            // Validate if the current backend type is CPU
            Assert.IsTrue(
                _worker.backendType == Unity.Sentis.BackendType.CPU,
                $"Invalid Sentis backend {_worker.backendType}, expected {BackendType.CPU}"
            );

            // Prepare the inputs for Sentis Inference
            PrepareSentisInferenceInputs();

            // Schedule the inference task on the worker
            _worker.Schedule(_inputTensors);

            if (_deformBackend == BackendType.GPU)
            {
                CommandBuffer cmd = CommandBufferPool.Get("Apply Deform GPU Block");
                ApplyDeformGPU(cmd);
                // Release the command buffer.
                CommandBufferPool.Release(cmd);
            }
            else
            {
                ApplyDeformCPU();
            }
        }

        /// <summary>
        /// Applies vertex deformation using CPU-side logic.
        ///
        /// This function processes the model outputs, which provide vertex delta values
        /// for each unique vertex, and accumulates the final deformed vertex positions.
        /// Optionally, deformation is modulated by an alpha mask if <see cref="_alphaMaskedDeformationWeightEnabled"/> is enabled.
        ///
        /// Key operations in this function include:
        /// - Preparing and resizing internal buffers: <see cref="_vertexDeltas"/>, <see cref="_deformedVertexPositions"/>.
        /// - Iterating through model output tensors from <see cref="_modelOutputs"/> and computing
        ///   vertex deltas per unique vertex using <see cref="Worker.PeekOutput()"/>.
        /// - Modulating deformation strength using <see cref="deformationWeight"/> and,
        ///   if enabled, <see cref="alphaMaskInfluenceWeight"/> derived from the vertex color alpha channel,
        ///   where color and vertex buffer resources are loaded using <see cref="LoadColorBufferResources()"/>.
        /// - Applying the final transformed vertex positions to the mesh via <see cref="meshInstance.SilentlySetVertices()"/>.
        ///
        /// Note:
        /// Alpha modulation is performed by extracting the high 8 bits (alpha channel)
        /// from the vertex color buffer <see cref="_colorBuffer"/> and blending with <see cref="alphaMaskInfluenceWeight"/>.
        /// </summary>
        void ApplyDeformCPU()
        {
            // Arrays to hold vertex deformation deltas and deformed vertex positions as final output
            int uniqueVertexCount = neuralDeformerDatasetMetaInfo.UniqueVertexCount;
            ArrayUtils.ResizeChecked(ref _vertexDeltas, uniqueVertexCount);
            ArrayUtils.ResizeChecked(ref _deformedVertexPositions, _meshBuffers.vertexCount);
            ArrayUtils.ClearChecked(_vertexDeltas);
            Array.Copy(
                _meshBuffers.vertexPositions,
                _deformedVertexPositions,
                _meshBuffers.vertexCount
            );

            // outputTensor contains a flatten vector3 delta values of all unique vertices
            // Aka: outputTensor = [vertex0_delta_x, vertex0_delta_y, vertex0_delta_z, vertex1_delta_x, vertex1_delta_y, vertex1_delta_y, ...]
            var outputTensor = _worker.PeekOutput(_modelOutputs[0].name) as Tensor<float>;
            using var flattenedUniqueVertexDeltas = outputTensor.DownloadToNativeArray();

            for (int uniqueVertexIndex = 0; uniqueVertexIndex < uniqueVertexCount; ++uniqueVertexIndex)
            {
                int outputTensorOffset = uniqueVertexIndex * 3;
                _vertexDeltas[uniqueVertexIndex] = new Vector3(
                    Denormalize(flattenedUniqueVertexDeltas[outputTensorOffset]),
                    Denormalize(flattenedUniqueVertexDeltas[outputTensorOffset + 1]),
                    Denormalize(flattenedUniqueVertexDeltas[outputTensorOffset + 2])
                );
            }

            // Apply the final vertex deltas to the mesh vertices
            // If alpha mask influence is enabled, apply the mask to adjust the deformation weight
            LoadColorBufferResources();
            if (_alphaMaskedDeformationWeightEnabled)
            {
                for (int vertexIndex = 0; vertexIndex < _meshBuffers.vertexCount; vertexIndex++)
                {
                    // Read the color at the vertex index and extract the alpha value as the mask weight
                    uint color = _colorBuffer[
                        vertexIndex * (_colorBufferStride / UnsafeUtility.SizeOf<uint>())
                            + _colorBufferOffset
                    ];
                    // Here, "color >> 24" shifts the color value right by 24 bits to extract the alpha channel value
                    // Then, "/ 255" Denormalizes the extracted alpah value to range [0, 1]
                    float alphaMaskWeight = math.saturate((color >> 24) / 255f);
                    // The default deformation weight is 1, if Alpha Mask is enabled as above
                    // The mask is extracted using the alpha component (high 8 bits) of the color and blended with the weight influence via math.lerp(...)
                    // See: https://docs.unity.cn/cn/tuanjiemanual/ScriptReference/Mathf.Lerp.html
                    float alphaMaskedDeformationWeight = math.lerp(
                        1f,
                        alphaMaskWeight,
                        math.saturate(alphaMaskInfluenceWeight)
                    );
                    // Apply the deformation weight considering the mask influence
                    _deformedVertexPositions[vertexIndex] +=
                        math.saturate(deformationWeight)
                        * alphaMaskedDeformationWeight
                        * _vertexDeltas[_vertexMapping[vertexIndex]];
                }
            }
            else
            {
                // If not alpha masked deformation weitght enabled, directly apply the deformation weight
                for (int vertexIndex = 0; vertexIndex < _meshBuffers.vertexCount; vertexIndex++)
                {
                    _deformedVertexPositions[vertexIndex] +=
                        math.saturate(deformationWeight) * _vertexDeltas[_vertexMapping[vertexIndex]];
                }
            }

            // Set the final deformed vertex positions to the mesh
            meshInstance.SilentlySetVertices(_deformedVertexPositions);
        }

        /// <summary>
        /// Resets the mesh to its original vertex positions without applying any deformation.
        /// This method is called when deformation is disabled to restore the original state of the mesh.
        /// </summary>
        void NoDeformCPU()
        {
            // Set the original (undeformed) vertex positions back to the mesh.
            meshInstance.SilentlySetVertices(_meshBuffers.vertexPositions);
        }

        /// <summary>
        /// Recalculates the normals of the mesh on the CPU when required.
        /// This method ensures the normals are updated if the deformation has been applied to the mesh, 
        /// and if the "recalculateNormals" flag is set to true, which is necessary after the mesh vertices have been modified.
        /// It uses Unity's built-in method to recalculate normals based on the modified mesh.
        /// </summary>
        void RecalculateNormalsCPU()
        {
            // Check if recalculating normals is enabled and if the skinned mesh renderer and its mesh are valid.
            if (
                !recalculateNormals // Ensure recalculating normals is enabled
                || _skinnedMeshRenderer == null // Ensure the skinned mesh renderer is initialized
                || _skinnedMeshRenderer.sharedMesh == null // Ensure the shared mesh is available
            )
                return;

            DebugLog("NeuralDeformerPlayer.RecalculateNormalsCPU()");

            // Use the target mesh buffer, which holds the vertex data for the mesh.
            using GraphicsBuffer targetMeshBuffer = _skinnedMeshRenderer.GetVertexBuffer();
            if (targetMeshBuffer == null)
            {
                return;
            }

            // Resize the CPU mesh buffer to match the vertex count and float buffer stride.
            ArrayUtils.ResizeChecked(
                ref _cpuMeshBuffer,
                _meshBuffers.vertexCount * _floatBufferStride
            );
            // Copy the vertex data from the target mesh buffer to the CPU mesh buffer.
            targetMeshBuffer.GetData(_cpuMeshBuffer);

            // Resize the CPU vertex positions array to match the vertex count.
            ArrayUtils.ResizeChecked(ref _cpuVertexPositions, _meshBuffers.vertexCount);
            // Extract vertex positions from the CPU mesh buffer and store them in the CPU vertex positions array.
            for (int vi = 0; vi < _meshBuffers.vertexCount; vi++)
            {
                var viPositionOffset = vi * _floatBufferStride + _floatPositionOffset;
                _cpuVertexPositions[vi] = new Vector3(
                    _cpuMeshBuffer[viPositionOffset],
                    _cpuMeshBuffer[viPositionOffset + 1],
                    _cpuMeshBuffer[viPositionOffset + 2]
                );
            }

            // Instantiate the baked mesh if it's not already instantiated.
            _bakedMesh ??= Instantiate(meshAsset);
            // Set the vertices of the baked mesh to the CPU vertex positions.
            _bakedMesh.SilentlySetVertices(_cpuVertexPositions);
            // Recalculate the normals of the baked mesh.
            _bakedMesh.SilentlyRecalculateNormals();

            // Initialize the CPU vertex normals array if it's not already initialized.
            _cpuVertexNormals ??= new();
            // Get the normals of the baked mesh and store them in the CPU vertex normals array.
            _bakedMesh.GetNormals(_cpuVertexNormals);

            int i = 0;
            // Iterate through the CPU vertex normals and update the CPU mesh buffer with the new normals.
            foreach (var normal in _cpuVertexNormals)
            {
                var iPositionOffset = i * _floatBufferStride + _floatNormalOffset;
                _cpuMeshBuffer[iPositionOffset] = normal.x;
                _cpuMeshBuffer[iPositionOffset + 1] = normal.y;
                _cpuMeshBuffer[iPositionOffset + 2] = normal.z;
                ++i;
            }

            // Set the updated CPU mesh buffer back to the target mesh buffer.
            targetMeshBuffer.SetData(_cpuMeshBuffer);
        }
        #endregion


        #region GPU Inference & Postprocessing
        /// <summary>
        /// Applies vertex deformation using the GPU compute backend.
        ///
        /// Unlike <see cref="DeformCPU()"/>, this method performs Sentis model inference and deformation entirely on the GPU
        /// using a <see cref="CommandBuffer"/> and <see cref="Unity.Sentis.BackendType.GPUCompute"/> backend.
        /// Reference: https://docs.unity.cn/Packages/com.unity.sentis@2.1/manual/use-command-buffer.html
        ///
        /// The rest of the deformation logic, including input preparation and backend selection,
        /// follows the same general flow as in <see cref="DeformCPU()"/>.
        /// </summary>
        void DeformGPU()
        {
            DebugLog("NeuralDeformerPlayer.DeformGPU()");

            // Validate that the backend is GPUCompute. If not, this function should not be used.
            Assert.IsTrue(
                _worker.backendType == Unity.Sentis.BackendType.GPUCompute,
                $"Invalid sentis backend {_worker.backendType}, expected {Unity.Sentis.BackendType.GPUCompute}"
            );

            // Create a CommandBuffer to dispatch GPU tasks.
            CommandBuffer cmd = CommandBufferPool.Get("Deform GPU Block");

            // Model Inference using Sentis via GPUCompute
            cmd.BeginSample("DeformGPU - SentisInference");

            cmd.BeginSample("DeformGPU - PrepareSentisInferenceInputs");

            // Prepare the inputs for Sentis Inference
            PrepareSentisInferenceInputs();

            cmd.EndSample("DeformGPU - PrepareSentisInferenceInputs");
            // Schedule the inference task on the worker
            cmd.ScheduleWorker(_worker, _inputTensors);
            cmd.EndSample("DeformGPU - SentisInference");

            if (_deformBackend == BackendType.GPU)
            {
                ApplyDeformGPU(cmd);
            }
            else
            {
                // Execute the command buffer to apply the deformation on the GPU.
                Graphics.ExecuteCommandBuffer(cmd);

                ApplyDeformCPU();
            }

            // Release the command buffer.
            CommandBufferPool.Release(cmd);
        }

        /// <summary>
        /// Applies vertex deformation using GPU compute shaders.
        ///
        /// This function dispatches compute shader kernels to process deformation
        /// deltas and apply them directly to GPU-side mesh vertex buffers.
        ///
        /// Steps include:
        /// - Loading color and vertex buffer resources using <see cref="LoadColorBufferResources()"/>.
        /// - Setting compute shader parameters for deformation using <see cref="deformationWeight"/> and
        ///   <see cref="alphaMaskInfluenceWeight"/>, along with normalization bounds from <see cref="NeuralDeformerDatasetMetaInfo"/>.
        /// - Binding relevant GPU buffers, including <see cref="_vertexPositionsBuffer"/>, <see cref="_vertexMappingBuffer"/>,
        ///   and <see cref="_gpuColorBuffer"/> (if alpha masking is used).
        /// - Iteratively dispatching the compute shader for each unique vertex,
        ///   using delta values from <see cref="Worker.PeekOutput()"/> and mapped via <see cref="ComputeTensorData.Pin()"/>.
        /// - Writing final deformed positions into the target mesh buffer obtained via <see cref="meshInstance.GetVertexBuffer()"/>.
        ///
        /// In editor mode and when GPU skinning is disabled, the method synchronizes the GPU results
        /// back to CPU-accessible buffers <see cref="_cpuMeshBuffer"/> and <see cref="_cpuVertexPositions"/>,
        /// updating <see cref="meshInstance"/> accordingly to ensure correctness in CPU skinning scenarios.
        /// </summary>
        void ApplyDeformGPU(CommandBuffer cmd)
        {
            // Apply Deformations using GPU Compute Shader
            cmd.BeginSample("Deform - ApplyDeform");

            // Use the target mesh buffer, which is the vertex buffer of the mesh.
            using GraphicsBuffer targetMeshBuffer = meshInstance.GetVertexBuffer(_bufferStream);

            // Load the color buffer resources for calculating alpha mask influence.
            LoadColorBufferResources();

            // Now, set all the shader parameters for applying deformation.
            // Set weights for deformation and alpha mask influence.
            cmd.SetComputeFloatParam(
                s_computeShader,
                Uniforms._deformationWeight,
                enableDeformation ? math.saturate(deformationWeight) : 0f
            );
            cmd.SetComputeFloatParam(
                s_computeShader,
                Uniforms._alphaMaskInfluenceWeight,
                math.saturate(alphaMaskInfluenceWeight)
            );
            // Set the model's normalization parameters, used to map the output deltas to the correct range.
            cmd.SetComputeFloatParam(
                s_computeShader,
                Uniforms._modelNormalizationMaxValue,
                neuralDeformerDatasetMetaInfo.MaxDelta
            );
            cmd.SetComputeFloatParam(
                s_computeShader,
                Uniforms._modelNormalizationMinValue,
                neuralDeformerDatasetMetaInfo.MinDelta
            );
            // Set the number of vertices in the mesh and the stride/offsets of the vertex buffer.
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexCount,
                _meshBuffers.vertexCount
            );
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexBufferStride,
                _byteBufferStride
            );
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexPositionAttributeOffset,
                _bytePositionOffset
            );
            // Set the unique vertex index mapping buffer, which holds the index mapping from true vertices to unique ones.
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ApplyDeformation,
                Uniforms._vertexMappingBuffer,
                _vertexMappingBuffer
            );
            // Set the vertex positions buffer for the original neutral pose positions.
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ApplyDeformation,
                Uniforms._vertexPositionsBuffer,
                _vertexPositionsBuffer
            );
            // Set the target mesh buffer, where the deformed vertices will be written.
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ApplyDeformation,
                Uniforms._targetMeshBufferRW,
                targetMeshBuffer
            );
            // If alpha mask influence is enabled, set the relevant color buffer parameters for it.
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._alphaMaskedDeformationWeightEnabled,
                _alphaMaskedDeformationWeightEnabled ? 1 : 0
            );
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ApplyDeformation,
                Uniforms._colorBuffer,
                _gpuColorBuffer
            );
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._colorBufferOffset,
                _colorBufferOffset
            );
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._colorBufferStride,
                _colorBufferStride
            );

            var outputTensor = _worker.PeekOutput(_modelOutputs[0].name) as Tensor<float>;
            // Pin the output tensor data for use in the compute shader.
            var deformationDeltaBuffer = ComputeTensorData.Pin(outputTensor).buffer;
            // Set the deformation delta buffer, which contains the deformation deltas for the mesh vertices.
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ApplyDeformation,
                Uniforms._deformationDeltaBuffer,
                deformationDeltaBuffer
            );

            // Dispatch the compute shader to apply the deformation.
            int vertexCount = _meshBuffers.vertexCount;
            int groupsX = GetDispatchGroupsCount(Kernels.ApplyDeformation, (uint)vertexCount);
            cmd.DispatchCompute(s_computeShader, Kernels.ApplyDeformation, groupsX, 1, 1);

            cmd.EndSample("Deform - ApplyDeform");

            // Execute the command buffer to apply the deformation on the GPU.
            Graphics.ExecuteCommandBuffer(cmd);
#if UNITY_EDITOR
            if (!PlayerSettings.gpuSkinning)
            {
                // If CPU skinning is used, we should manually synchronize deformation to the CPU buffer,
                // from which CPU skinning calculations will fetch vertex data.
                ArrayUtils.ResizeChecked(
                    ref _cpuMeshBuffer,
                    _meshBuffers.vertexCount * _floatBufferStride
                );
                targetMeshBuffer.GetData(_cpuMeshBuffer);

                ArrayUtils.ResizeChecked(ref _cpuVertexPositions, _meshBuffers.vertexCount);
                for (int vi = 0; vi < _meshBuffers.vertexCount; vi++)
                {
                    var viPositionOffset = vi * _floatBufferStride + _floatPositionOffset;
                    _cpuVertexPositions[vi] = new Vector3(
                        _cpuMeshBuffer[viPositionOffset],
                        _cpuMeshBuffer[viPositionOffset + 1],
                        _cpuMeshBuffer[viPositionOffset + 2]
                    );
                }

                meshInstance.SilentlySetVertices(_cpuVertexPositions);
            }
#endif
        }

        /// <summary>
        /// Recalculates the normals of the mesh using GPU Compute Shader,
        /// which applies the normal recalculation based on the mesh's triangle adjacency data.
        /// This method performs two main tasks:
        /// 1. Computes the triangle normals based on the mesh's geometry using compute shaders.
        /// 2. Computes the vertex normals by averaging the normals of adjacent triangles and
        ///    then applying the result to the mesh.
        /// </summary>
        void RecalculateNormalsGPU()
        {
            // Check if recalculating normals is enabled and if the skinned mesh renderer and its mesh are valid.
            if (
                !recalculateNormals // Ensure recalculating normals is enabled
                || _skinnedMeshRenderer == null // Ensure the skinned mesh renderer is initialized
                || _skinnedMeshRenderer.sharedMesh == null // Ensure the shared mesh is available
                || _triangleNormalBuffer == null
                || !_triangleNormalBuffer.IsValid()
            )
                return;

            DebugLog("NeuralDeformerPlayer.RecalculateNormalsGPU()");

            // Create a command buffer for dispatching GPU tasks.
            CommandBuffer cmd = CommandBufferPool.Get("Recalculate Normals GPU Block");

            cmd.BeginSample("RecalculateNormalsGPU - RecalculateNormals");
            // Use the target mesh buffer, which holds the vertex data for the mesh.
            using GraphicsBuffer targetMeshBuffer = _skinnedMeshRenderer.GetVertexBuffer();
            if (targetMeshBuffer == null)
            {
                return;
            }

            // ========================================
            // Step 1: Compute All the Triangle Normals
            // ========================================
            cmd.BeginSample("RecalculateNormalsGPU - ComputeTriangleNormals");
            // Set the number of triangles in the mesh to the compute shader.
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._triangleCount,
                _meshAdjacency.triangleCount
            );
            // Set the vertex buffer stride, which is the size of each vertex in the buffer.
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexBufferStride,
                _byteBufferStride
            );
            // Set the vertex position attribute offset (for accessing position data in the buffer).
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexPositionAttributeOffset,
                _bytePositionOffset
            );
            // Set the compute shader's buffers for triangle index data
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeTriangleNormals,
                Uniforms._meshTriangleIndexBuffer,
                _meshTriangleIndexBuffer
            );
#if UNITY_EDITOR
            // Set the mesh buffer to get normals of vertices to calculate triangle noramls
            if (PlayerSettings.gpuSkinning)
            {
                cmd.SetComputeBufferParam(
                    s_computeShader,
                    Kernels.ComputeTriangleNormals,
                    Uniforms._targetMeshBufferRW,
                    targetMeshBuffer
                );
            }
            else
            {
                ArrayUtils.ResizeChecked(
                    ref _cpuMeshBuffer,
                    _meshBuffers.vertexCount * _floatBufferStride
                );
                targetMeshBuffer.GetData(_cpuMeshBuffer);
                _gpuTempMeshBuffer.SetData(_cpuMeshBuffer);

                cmd.SetComputeBufferParam(
                    s_computeShader,
                    Kernels.ComputeTriangleNormals,
                    Uniforms._targetMeshBufferRW,
                    _gpuTempMeshBuffer
                );
            }
#else
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeTriangleNormals,
                Uniforms._targetMeshBufferRW,
                targetMeshBuffer
            );
#endif

            // Set the ReadWrite buffer which stores the calculated triangle noramls,
            // which will be used in the next step.
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeTriangleNormals,
                Uniforms._triangleNormalBufferRW,
                _triangleNormalBuffer
            );

            // Dispatch the compute shader for triangle normal calculation.
            int groupsXForComputeTriangleNormals = GetDispatchGroupsCount(
                Kernels.ComputeTriangleNormals,
                (uint)_meshAdjacency.triangleCount
            );
            cmd.DispatchCompute(
                s_computeShader,
                Kernels.ComputeTriangleNormals,
                groupsXForComputeTriangleNormals,
                1, 1
            );

            cmd.EndSample("RecalculateNormalsGPU - ComputeTriangleNormals");

            // ========================================
            // Step 2: Compute Vertex Normals and Apply
            // ========================================
            cmd.BeginSample("RecalculateNormalsGPU - ComputeVertexNormals");
            // Set the number of vertices in the mesh to the compute shader.
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexCount,
                _meshBuffers.vertexCount
            );
            // Set the vertex buffer stride and normal attribute offset.
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexBufferStride,
                _byteBufferStride
            );
            cmd.SetComputeIntParam(
                s_computeShader,
                Uniforms._vertexNormalAttributeOffset,
                _byteNormalOffset
            );
            // Set the ReadOnly triangle noramls buffer calculated in the previous step
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeVertexNormals,
                Uniforms._triangleNormalBufferRO,
                _triangleNormalBuffer
            );
            // Set the adjacency data (triangle index buffer, and strides/offsets)
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeVertexNormals,
                Uniforms._meshAdjacencyTriangleIndexBuffer,
                _meshAdjacencyTriangleIndexBuffer
            );
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeVertexNormals,
                Uniforms._meshAdjacencyTriangleIndexStrideBuffer,
                _meshAdjacencyTriangleIndexStrideBuffer
            );
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeVertexNormals,
                Uniforms._meshAdjacencyTriangleIndexOffsetBuffer,
                _meshAdjacencyTriangleIndexOffsetBuffer
            );

#if UNITY_EDITOR
            // Set the target mesh buffer, where the recalculated vertex normals will be written.
            if (PlayerSettings.gpuSkinning)
            {
                cmd.SetComputeBufferParam(
                    s_computeShader,
                    Kernels.ComputeVertexNormals,
                    Uniforms._targetMeshBufferRW,
                    targetMeshBuffer
                );
            }
            else
            {
                cmd.SetComputeBufferParam(
                    s_computeShader,
                    Kernels.ComputeVertexNormals,
                    Uniforms._targetMeshBufferRW,
                    _gpuTempMeshBuffer
                );
            }
#else
            cmd.SetComputeBufferParam(
                s_computeShader,
                Kernels.ComputeVertexNormals,
                Uniforms._targetMeshBufferRW,
                targetMeshBuffer
            );
#endif

            // Dispatch the compute shader to calculate vertex normals based on the triangle normals.
            int groupsXForComputeVertexNormals = GetDispatchGroupsCount(
                Kernels.ComputeVertexNormals,
                (uint)_meshAdjacency.vertexCount
            );
            cmd.DispatchCompute(
                s_computeShader,
                Kernels.ComputeVertexNormals,
                groupsXForComputeVertexNormals,
                1, 1
            );

            cmd.EndSample("RecalculateNormalsGPU - ComputeVertexNormals");
            cmd.EndSample("RecalculateNormalsGPU - RecalculateNormals");

            // Execute the command buffer to apply the recalculated normals on the GPU.
            Graphics.ExecuteCommandBuffer(cmd);

            // Release the command buffer.
            CommandBufferPool.Release(cmd);
#if UNITY_EDITOR
            if (!PlayerSettings.gpuSkinning)
            {
                // If CPU skinning is used, we should manually synchronize normals to the CPU buffer,
                // from which CPU skinning calculations will fetch vertex data.
                ArrayUtils.ResizeChecked(
                    ref _cpuMeshBuffer,
                    _meshBuffers.vertexCount * _floatBufferStride
                );
                _gpuTempMeshBuffer.GetData(_cpuMeshBuffer);
                targetMeshBuffer.SetData(_cpuMeshBuffer);
            }
#endif
        }
        #endregion


        #region Utilities
        /// <summary>
        /// Checks whether the necessary resources and configurations are available to perform mesh deformation.
        /// If any of these resources are missing or invalid, deformation will not be applied, and an error message will be logged.
        /// </summary>
        /// <returns>
        /// Returns `true` if all required resources and configurations are available for deformation, 
        /// otherwise returns `false`. This helps in ensuring that the deformation process only occurs when 
        /// all necessary data and resources are properly set up.
        /// </returns>
        private bool CheckDeformationAvailable()
        {
            bool result = true;

            if (neuralDeformerDatasetMetaInfo == null)
            {
                Debug.LogError(
                    "Resource Not Available: `neuralDeformerDatasetMetaInfo` is null, please provide the Deformer Data Meta Info"
                );
                result = false;
            }
            if (modelAsset == null)
            {
                Debug.LogError(
                    "Resource Not Available: `modelAsset` is null, please provide the `.onnx` or `.sentis` Model Asset"
                );
                result = false;
            }
            if (meshAsset == null)
            {
                Debug.LogError("Resource Not Available: `meshAsset` is invalid");
                result = false;
            }

            return result;
        }

        /// <summary>
        /// Converts a DynamicTensorShape (which can have dynamic dimensions) into a static TensorShape.
        /// The function ensures that the shape is properly translated, and if some dimensions are dynamic (i.e., not defined),
        /// they are replaced with default values. This is necessary because the model inputs typically require a static shape 
        /// for processing, while DynamicTensorShape may allow some dimensions to remain undefined at runtime.
        /// </summary>
        /// <param name="shape">
        /// The input DynamicTensorShape that needs to be converted into a static TensorShape.
        /// </param>
        /// <returns>
        /// Returns a TensorShape object with the dimensions of the input DynamicTensorShape, where any dynamic dimensions 
        /// (denoted as -1) are replaced with default values (ones), except those that are statically defined.
        /// </returns>
        private TensorShape ConvertToTensorShape(DynamicTensorShape shape)
        {
            Assert.IsTrue(shape != null, "DynamicTensorShape is null");
            Assert.IsTrue(!shape.isRankDynamic, "DynamicTensorShape has no rank");
            if (shape.IsStatic())
            {
                return shape.ToTensorShape();
            }
            else
            {
                var shapeValues = shape.ToIntArray();
                var shapeOutput = TensorShape.Ones(shape.rank);
                for (int i = 0; i < shape.rank; i++)
                {
                    if (shapeValues[i] != -1)
                    {
                        shapeOutput[i] = shapeValues[i];
                    }
                }
                return shapeOutput;
            }
        }

        /// <summary>
        /// Frees (disposes of) a disposable object, ensuring proper cleanup of resources. 
        /// This method checks if the object is valid and implements the IDisposable interface, and if so, it disposes of it. 
        /// It is used to release resources, both in the case of individual objects and lists of disposable items.
        /// </summary>
        /// <param name="field">
        /// The <see cref="FieldInfo"/> representing the field holding the disposable object to be disposed. 
        /// This field should contain either a single disposable object or a list of disposable objects.
        /// </param>
        /// <param name="isList">
        /// A boolean flag indicating whether the field contains a list of disposable objects. If this is `true`, 
        /// the method will iterate over the list and dispose of each individual object. If `false`, it will dispose of a single object.
        /// </param>
        private void FreeDisposableObject(FieldInfo field, bool isList = false)
        {
            var obj = field.GetValue(this);
            var objName = field.Name;
            DebugLog(
                $"NeuralDeformerPlayer.FreeDisposableObject(): object `{objName}` start disposing"
            );

            if (obj == null)
            {
                if (_initialized && _enableDeformationDebugLogging)
                {
                    DebugLog(
                        $"NeuralDeformerPlayer.FreeDisposableObject(): object `{objName}` is null, ignored"
                    );
                }
                return;
            }

            if (isList && obj is System.Collections.IEnumerable listObj)
            {
                foreach (var item in listObj)
                {
                    try
                    {
                        if (item is IDisposable disposableObject)
                        {
                            disposableObject?.Dispose();
                        }
                    }
                    catch (Exception ex)
                    {
                        Debug.LogError($"Dispose error in `{objName}` item: {ex}");
                    }
                }
            }
            else if (obj is IDisposable disposableObject)
            {
                disposableObject?.Dispose();
            }
            else
            {
                Debug.LogWarning(
                    $"Object `{objName}` of type {obj.GetType()} does not implement IDisposable, ignored"
                );
                return;
            }
            field.SetValue(this, null);

            DebugLog(
                $"NeuralDeformerPlayer.FreeDisposableObject(): object `{objName}` disposed"
            );
        }

        /// <summary>
        /// Frees all allocated disposable objects used by the NeuralDeformerPlayer instance.
        /// This method iterates through all fields that are marked as disposable and disposes of them properly.
        /// It ensures that any dynamically allocated resources, such as memory or GPU resources, are released
        /// when the deformation process is no longer needed. This helps to avoid memory leaks and unnecessary
        /// resource consumption during the object's lifecycle.
        /// </summary>
        private void FreeAllDisposableObjects()
        {
            if (
                _disposableFields == null
                || _listDisposableFields == null
                || _gpuResourceDisposableFields == null
            )
            {
                InitializeAllDisposableFields();
            }
            foreach (var field in _disposableFields)
            {
                FreeDisposableObject(field, false);
            }
            foreach (var field in _listDisposableFields)
            {
                FreeDisposableObject(field, true);
            }
            foreach (var field in _gpuResourceDisposableFields)
            {
                FreeDisposableObject(field, false);
            }
        }

        /// <summary>
        /// Initializes all disposable fields in the <see cref="NeuralDeformerPlayer"/> class.
        /// This function iterates through all member variables of the class and identifies
        /// which fields need to be disposed of when the object is destroyed or deactivated.
        /// It categorizes the fields into disposable types and stores them in specific lists:
        ///   - <see cref="_disposableFields"/>
        ///   - <see cref="_listDisposableFields"/>
        ///   - <see cref="_gpuResourceDisposableFields"/> 
        /// These fields are marked with custom attributes to indicate they should be disposed
        /// of when no longer needed. This function is called when setting up the resource cleanup
        /// process to ensure that memory and GPU resources are properly managed.
        /// </summary>
        private void InitializeAllDisposableFields()
        {
            _disposableFields = new List<FieldInfo>();
            _listDisposableFields = new List<FieldInfo>();
            _gpuResourceDisposableFields = new List<FieldInfo>();
            _gpuInferenceResourceDisposableFields = new List<FieldInfo>();
            _gpuPostprocessingResourceDisposableFields = new List<FieldInfo>();
            // Iterate through all member variables in the class
            FieldInfo[] fields = GetType()
                .GetFields(BindingFlags.NonPublic | BindingFlags.Instance);
            foreach (var field in fields)
            {
                // Check if the field has the DisposeOnDestroy / ListDisposeOnDestroyAttribute attribute
                if (Attribute.IsDefined(field, typeof(DisposeOnDestroyAttribute)))
                {
                    _disposableFields.Add(field);
                }
                else if (Attribute.IsDefined(field, typeof(ListDisposeOnDestroyAttribute)))
                {
                    _listDisposableFields.Add(field);
                }
                else if (Attribute.IsDefined(field, typeof(GPUResourceDisposeOnDestroyAttribute)))
                {
                    _gpuResourceDisposableFields.Add(field);
                    if (
                        Attribute.IsDefined(
                            field,
                            typeof(GPUInferenceResourceDisposeOnDestroyAttribute)
                        )
                    )
                    {
                        _gpuInferenceResourceDisposableFields.Add(field);
                    }
                    else if (
                        Attribute.IsDefined(
                            field,
                            typeof(GPUPostprocessingResourceDisposeOnDestroyAttribute)
                        )
                    )
                    {
                        _gpuPostprocessingResourceDisposableFields.Add(field);
                    }
                }
            }
        }

        /// <summary>
        /// Denormalizes a value that has been normalized between -1 and 1 back to its original range
        /// defined by the minimum and maximum delta values specified in the <see cref="NeuralDeformerDatasetMetaInfo"/>.
        /// The purpose of this function is to map the normalized deformation values back to their original scale
        /// as required for mesh deformation, using the min and max delta values from the deformation data metadata.
        /// 
        /// The denormalization process works by mapping the normalized value back to the original range. 
        /// The formula used for denormalization is as follows:
        /// 
        ///     `denormalized_value = Lerp(MinDelta, MaxDelta, (x * 0.5f) + 0.5f)`
        /// 
        /// Where:
        /// - `Lerp(MinDelta, MaxDelta, t)` is a linear interpolation function that computes a value between `MinDelta`
        ///   and `MaxDelta` based on the parameter `t`, which should be a value between 0 and 1.
        /// - `x` is the normalized value between -1 and 1.
        /// - `(x * 0.5f) + 0.5f` transforms the normalized range [-1, 1] into the range [0, 1], which is required by the `Lerp` function.
        /// 
        /// This formula ensures that when `x = -1`, the result will be `MinDelta`, and when `x = 1`, the result will be `MaxDelta`.
        /// The intermediate values of `x` will map proportionally between the two deltas.
        /// </summary>
        /// <param name="x">
        /// The normalized value to be denormalized. This value is typically between -1 and 1,
        /// representing a deformation delta that has been normalized to fit within this range.
        /// </param>
        /// <returns>
        /// Returns the denormalized value, which is mapped back to the range defined by
        /// <see cref="NeuralDeformerDatasetMetaInfo.MinDelta"/> and <see cref="NeuralDeformerDatasetMetaInfo.MaxDelta"/>.
        /// This value is used to apply the correct deformation deltas to the mesh.
        /// </returns>
        private float Denormalize(float x)
        {
            return math.lerp(
                neuralDeformerDatasetMetaInfo.MinDelta,
                neuralDeformerDatasetMetaInfo.MaxDelta,
                (x * 0.5f) + 0.5f
            );
        }

        /// <summary>
        /// Prepares input tensor data for Sentis model inference based on current joint rotations.
        ///
        /// This method writes joint rotation (quaternion) values into the model's input tensor,
        /// adapting the format based on the selected inference backend:
        ///
        /// - For <see cref="BackendType.GPU"/>:
        ///   Uses a temporary <see cref="NativeArray{T}"/> to copy joint quaternion data
        ///   and uploads it to the backend tensor via <see cref="Tensor{T}.dataOnBackend.Upload"/>.
        /// - For <see cref="BackendType.CPU"/>:
        ///   Writes the joint data directly to the tensor after ensuring any pending operations
        ///   are completed with <see cref="Tensor{T}.CompleteAllPendingOperations()"/>.
        ///
        /// Validates that the number of joints matches the expected tensor input shape
        /// and that the input tensor exists and has the expected rank and structure.
        /// </summary>
        private void PrepareSentisInferenceInputs()
        {
            // Ensure the number of model inputs is 1
            Assert.IsTrue(
                _modelInputs.Count == 1,
                $"Number of model input must be 1, but got {_modelInputs.Count}"
            );

            // Get the input tensor
            var inputTensor = _inputTensors[0] as Tensor<float>;
            Assert.IsNotNull(inputTensor);
            // Ensure the input tensor's rank is 3 and the number of joints matches the expected tensor input shape
            Assert.IsTrue(
                inputTensor.shape.rank == 3 && _joints.Count == inputTensor.shape.ToArray()[1],
                "Number of joints and number of inputs mismatch"
            );

            if (_sentisInferenceBackend == BackendType.GPU)
            {
                // For GPU, create a temporary array to store joint rotation data for GPU processing.
                var temp = new NativeArray<float>(_joints.Count * 4, Allocator.TempJob);
                for (int i = 0, j = 0; j < _joints.Count; ++j)
                {
                    temp[i++] = _joints[j].localRotation.x;
                    temp[i++] = _joints[j].localRotation.y;
                    temp[i++] = _joints[j].localRotation.z;
                    temp[i++] = _joints[j].localRotation.w;
                }
                // Upload the temporary array to the backend tensor
                inputTensor.dataOnBackend.Upload(temp, _joints.Count * 4);
                temp.Dispose();
            }
            else
            {
                // For CPU, first ensures all pending operations are completed
                inputTensor.CompleteAllPendingOperations();
                for (int j = 0; j < _joints.Count; ++j)
                {
                    inputTensor[0, j, 0] = _joints[j].localRotation.x;
                    inputTensor[0, j, 1] = _joints[j].localRotation.y;
                    inputTensor[0, j, 2] = _joints[j].localRotation.z;
                    inputTensor[0, j, 3] = _joints[j].localRotation.w;
                }
            }
        }

        /// <summary>
        /// Calculates the number of dispatch groups needed for a compute shader based on the kernel's thread group size and
        /// the total thread count.
        /// This function is used to determine how many thread groups should be dispatched to the GPU for a particular kernel
        /// execution in the compute shader.
        /// It ensures that the compute shader operates on the correct number of thread groups according to the workload size.
        ///
        /// The number of thread groups is determined by dividing the total number of threads by the number of threads per group.
        /// If the division results in a remainder, an additional group is dispatched to account for the leftover threads.
        ///</summary>
        /// <param name="kernel">The kernel ID of the compute shader, used to retrieve the kernel's thread group size.</param>
        /// <param name="threadCount">The total number of threads that need to be processed by the compute shader.</param>
        /// <returns>Returns the number of dispatch groups required to process all threads.</returns>
        static int GetDispatchGroupsCount(int kernel, uint threadCount)
        {
            s_computeShader.GetKernelThreadGroupSizes(
                kernel,
                out var groupX,
                out var groupY,
                out var groupZ
            );
            return (int)((threadCount + groupX - 1) / groupX);
        }

        /// <summary>
        /// Logs the execution time of a given action to the console, useful for profiling performance.
        /// This function measures the time it takes to execute a specific action (method), and logs it
        /// in milliseconds if `enableDeformationDebuging` is enabled. It uses Unity's Profiler to
        /// track and measure the time taken by the action.
        /// </summary>
        /// <param name="action">The action (method) whose performance will be logged.</param>
        private void LogPerformance(Action action)
        {
            if (enableDeformationDebuging)
            {
                string sectionName = action.Method.Name;
                float startTime = Time.realtimeSinceStartup;
                Profiler.BeginSample(sectionName);
                action.Invoke();
                Profiler.EndSample();
                float elapsedTime = Time.realtimeSinceStartup - startTime;
                DebugLog($"[Deformer Performance] {sectionName} took: {elapsedTime * 1000f} ms");
            }
            else
            {
                action.Invoke();
            }
        }

        /// <summary>
        /// Logs a message to the console if the deformation debug logging is enabled.
        /// This function is used to print debug messages related to the deformation process when 
        /// the `_enableDeformationDebugLogging` flag is set to `true`.
        /// </summary>
        /// <param name="message">The message to log to the console.</param>
        private void DebugLog(string message)
        {
            if (_enableDeformationDebugLogging)
            {
                Debug.Log(message);
            }
        }
        #endregion
    }
}
