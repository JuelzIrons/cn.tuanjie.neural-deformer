using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;


#if UNITY_EDITOR
using UnityEngine.Formats.Alembic.Importer;
#endif

namespace Tuanjie.NeuralDeformer
{
    /// <summary>
    /// <see cref="NeuralDeformerDatasetBuilder"/> class manages the process of generating dataset for deformer training.
    /// </summary>
    [ExecuteInEditMode]
    [AddComponentMenu("Neural Deformer/Neural Deformer Dataset Builder")]
    public class NeuralDeformerDatasetBuilder : MonoBehaviour
    {
        [Header("Task Specification")]
        [Tooltip("The avatar (character) GameObject driven by animations.")]
        public GameObject avatar;

        [Tooltip("The target GameObject with SkinnedMeshRenderer component for complicated deformation.")]
        public GameObject target;

        [Tooltip("The root bone (joint) of the character skeleton.")]
        public Transform rootBone;

        [Tooltip("The selected critical joints which will be used for deformation inference.")]
        public List<Transform> joints = new();

        [Tooltip("If false, vertex matching between skinned and deformed meshes is achieved by finding the nearest neighbor at the neutral pose. If true, users need to provide Alembic objects of skinned and deformed meshes at the neutral pose as the basis for vertex matching, which will yield better matching results.")]
        public bool useAlembicForVertexMatching;

        [Header("Unique Vertex Matching At Neutral Pose")]
        [Tooltip("The alembic object of the skinned mesh at neutral pose.")]
        public GameObject skinnedAlembic;

        [Tooltip("The alembic object of the deformed mesh at neutral pose.")]
        public GameObject deformAlembic;

        [Header("Pose Alignment")]
        [Tooltip("If true, the skinned mesh will be decoupled from animation to extract pure deformation, which will be converted to a neutral pose for vertex deltas calculation and enables visualization of the neutral-space mesh. If false, the input Alembic mesh must be a deformed neutral pose (no skeletal animation, only deformation).")]
        public bool extractPureDeformation;

        [Header("Path Settings")]
        [Tooltip("The directory containing AnimationClip files for poses.")]
        public string posesDir;

        [Tooltip("The directory containing Alembic files for outfit deformation.")]
        public string deformDir;

        [Tooltip("The directory to export the generated dataset. You can export to a path either within or outside of this project.")]
        public string datasetDir;

        [Tooltip("The task name.")]
        public string taskName;

#if UNITY_EDITOR
        #region Configurations

        /// <summary>
        /// The perturbation magnitude for calculating the vertex deltas with local coordinate systems. Should be a small positive number.
        /// </summary>
        private const float k_PerturbationEpsilon = 1e-3f;

        private bool m_Initialized = false;

        #endregion

        #region Input
        void OnValidate()
        {
            if (m_Initialized)
                return;

            // Execute when this component is first attached to a GameObject in edit mode.
            m_Initialized = true;
            // avatar = gameObject;
        }

        /// <summary>
        /// Unload current parameters
        /// </summary>
        private void UnLoad()
        {
            avatar = null;
            target = null;
            rootBone = null;
            joints.Clear();
            useAlembicForVertexMatching = false;
            skinnedAlembic = null;
            deformAlembic = null;
            extractPureDeformation = false;
            posesDir = null;
            deformDir = null;
            datasetDir = null;
            taskName = null;
        }

        /// <summary>
        /// Tool function to find child GameObject by name recursively.
        /// </summary>
        /// <param name="parent">Parent GameObject.</param>
        /// <param name="name">Name of target child GameObject.</param>
        /// <returns></returns>
        private static GameObject FindChildByNameRecursive(GameObject parent, string name)
        {
            if (parent == null || parent.name == name)
                return parent;

            foreach (Transform child in parent.transform)
            {
                var found = FindChildByNameRecursive(child.gameObject, name);
                if (found != null)
                    return found;
            }
            return null;
        }

        /// <summary>
        /// Try validating the parameters.
        /// </summary>
        /// <param name="animation">The result <see cref="Animation"/> component from <see cref="character"/>.</param>
        /// <param name="neutralMesh">The result outfit <see cref="Mesh"/> in default skinned pose from <see cref="outfit"/>.</param>
        /// <param name="posesPaths">The result paths where files containing <see cref="AnimationClip"/> prefabs are located.</param>
        /// <param name="deformPaths">The result paths where alembic files containing simulated outfits are located. </param>
        /// <returns>(<see cref="bool"/>) If the parameters are valid.</returns>
        public bool TryValidate(out Mesh neutralMesh, out string[] posesPaths, out string[] deformPaths)
        {
            neutralMesh = null;
            posesPaths = null;
            deformPaths = null;

            try
            {
                Assert.IsNotNull(avatar, "Character object can not be null.");
                Assert.IsNotNull(target, "Outfit object can not be null");
                var smr = target.GetComponent<SkinnedMeshRenderer>();
                Assert.IsNotNull(smr, $"SkinnedMeshRenderer component not found in outfit {target.name}.");
                neutralMesh = smr.sharedMesh;

                Assert.IsTrue(joints != null && joints.Count > 0, $"Joints can not be null or empty.");
                for (int i = 0; i < joints.Count; i++)
                    Assert.IsNotNull(joints[i], $"Joint {i} can not be null.");

                if (useAlembicForVertexMatching)
                {
                    Assert.IsNotNull(skinnedAlembic, "Skinned Alembic object can not be null.");
                    Assert.IsNotNull(deformAlembic, "Deformed Alembic object can not be null.");    
                }        

                Assert.IsFalse(string.IsNullOrEmpty(posesDir), "Poses directory can not be null or empty.");
                var posesGuids = AssetDatabase.FindAssets("t:AnimationClip", new[] { posesDir });
                Assert.IsTrue(posesGuids.Length > 0, $"No AnimationClip found in {posesDir}.");

                Assert.IsFalse(string.IsNullOrEmpty(deformDir), "Deform directory can not be null or empty.");
                var deformGuids = AssetDatabase.FindAssets("t:GameObject", new[] { deformDir });
                Assert.IsTrue(deformGuids.Length > 0, $"No alembic file found in {deformDir}.");
                Assert.IsTrue(deformGuids.Length == posesGuids.Length, $"Number of alembic files in {deformDir} does not match number of AnimationClips in {posesDir}.");

                // Batch-convert asset GUIDs to path strings
                posesPaths = Array.ConvertAll(posesGuids, AssetDatabase.GUIDToAssetPath);
                deformPaths = Array.ConvertAll(deformGuids, AssetDatabase.GUIDToAssetPath);

                Assert.IsFalse(string.IsNullOrEmpty(datasetDir), "Export directory can not be null or empty.");
                Assert.IsFalse(string.IsNullOrEmpty(taskName), "Task name can not be null or empty.");
            }
            catch (AssertionException e)
            {
                string message = e.Message.Split('\n')[0];
                EditorUtility.DisplayDialog("Validate", message, "OK");
                return false;
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Validate", e.Message, "OK");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Try getting pose clip from given file path.
        /// </summary>
        /// <param name="posePath">File path.</param>
        /// <param name="name">The result pose name.</param>
        /// <param name="clip">The result <see cref="AnimationClip"/> component.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private bool TryGetPoseClip(string posePath, out string name, out AnimationClip clip)
        {
            name = null;
            clip = null;

            try
            {
                var go = AssetDatabase.LoadAssetAtPath<GameObject>(posePath);
                Assert.IsNotNull(go, $"GameObject not found at {posePath}.");
                name = go.name;

                clip = AssetDatabase.LoadAssetAtPath<AnimationClip>(posePath);
                Assert.IsNotNull(clip, $"AnimationClip not found at {posePath}.");
                
            }
            catch (AssertionException e)
            {
                string message = e.Message.Split('\n')[0];
                EditorUtility.DisplayDialog("Pose Sample", message, "OK");
                return false;
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Pose Sample", e.Message, "OK");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Try getting deformed outfit data from given file path.
        /// </summary>
        /// <param name="deformPath">File path.</param>
        /// <param name="instance">The result outfit <see cref="GameObject"/> which is already loaded into the current scene.</param>
        /// <param name="mesh">The result deformed outfit <see cref="Mesh"/>.</param>
        /// <param name="player">The result <see cref="AlembicStreamPlayer"/> component which controls the animation timeline of <paramref name="instance"/>.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private bool TryGetDeformInstance(string deformPath, out GameObject instance, out Mesh mesh, out AlembicStreamPlayer player)
        {
            instance = null;
            mesh = null;
            player = null;

            try
            {
                var go = AssetDatabase.LoadAssetAtPath<GameObject>(deformPath);
                Assert.IsNotNull(go, $"GameObject not found at {deformPath}.");
                Assert.IsNotNull(go.GetComponentInChildren<MeshFilter>(), $"MeshFilter component not found in {go.name}'s children.");
                Assert.IsNotNull(go.GetComponentInChildren<AlembicStreamPlayer>(), $"AlembicStreamPlayer component not found in {go.name}'s children.");

                // Instantiate the prefab in the scene, or any animation will not work.
                instance = Instantiate(go);

                var mf = instance.GetComponentInChildren<MeshFilter>();
                // The mesh will deform correspondingly with animation of the host GameObject.
                mesh = mf.sharedMesh;
                player = instance.GetComponent<AlembicStreamPlayer>();
            }
            catch (AssertionException e)
            {
                string message = e.Message.Split('\n')[0];
                EditorUtility.DisplayDialog("Deform Sample", message, "OK");
                return false;
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Deform Sample", e.Message, "OK");
                return false;
            }

            return true;
        }

        #endregion

        #region Process
        /// <summary>
        /// Try getting unique vertices from skinned mesh, and return the unique vertex indices and a vertex table mapping.
        /// </summary>
        /// <param name="mesh">The target skinned mesh.</param>
        /// <param name="uniqueIndices">The index array of vertices with unique position.</param>
        /// <param name="vertexTable">The index mapping from true vertices to reduced ones.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private void GetUniqueVertices(Mesh mesh, out int[] uniqueIndices, out int[] vertexTable)
        {
            var vertices = mesh.vertices;
            var uniqueCount = 0;
            Dictionary<Vector3, int> vertexDict = new();

            List<int> uniqueIndicesList = new();
            vertexTable = new int[vertices.Length];

            for (int i = 0, n = vertices.Length; i < n; i++)
            {
                var v = vertices[i];
                if (!vertexDict.TryGetValue(v, out int uniqueId))
                {
                    uniqueId = uniqueCount++;
                    vertexDict.Add(v, uniqueId);

                    uniqueIndicesList.Add(i);
                }

                vertexTable[i] = uniqueId;
            }

            uniqueIndices = uniqueIndicesList.ToArray();
            Debug.Log($"Unique vertices of {mesh.name}: {uniqueCount}");
        }

        /// <summary>
        /// Try matching unique vertices at neutral pose between skinned mesh and deformed mesh.
        /// Vertices at the same index of <paramref name="skinnedVerts"/> and <paramref name="deformVerts"/> are guaranteed to be matched.
        /// </summary>
        /// <param name="skinnedVerts">The unique vertices of skinned mesh at neutral pose.</param>
        /// <param name="deformVerts">The unique vertices of deformed mesh at neutral pose.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private bool TryMatchUniqueVerticesAtNeutralPose(out Vector3[] skinnedVerts, out Vector3[] deformVerts)
        {
            skinnedVerts = null;
            deformVerts = null;

            string skinnedAlembicPath = AssetDatabase.GetAssetPath(skinnedAlembic);
            string deformAlembicPath = AssetDatabase.GetAssetPath(deformAlembic);
            if (!TryGetDeformInstance(skinnedAlembicPath, out var skinnedNeutralInstance, out var skinnedNeutralMesh, out var skinnedNeutralPlayer) ||
                !TryGetDeformInstance(deformAlembicPath, out var deformNeutralInstance, out var deformNeutralMesh, out var deformNeutralPlayer))
            {
                return false;
            }

            skinnedNeutralPlayer.UpdateImmediately(0.0f);
            deformNeutralPlayer.UpdateImmediately(0.0f);

            skinnedVerts = skinnedNeutralMesh.vertices;
            deformVerts = deformNeutralMesh.vertices;
            if (skinnedVerts.Length != deformVerts.Length)
            {
                EditorUtility.DisplayDialog("Vertex Mapping",
                    $"The vertex numbers of Skinned Alembic and Deform Alembic are expected to be equal, " +
                    $"but got {skinnedVerts.Length} and {deformVerts.Length}." +
                    $"Please check whether the Alembic files contain irrelevant information, such as normals and UVs.",
                    "OK");
                return false;
            }

            var maxDistance = 0.0f;
            var avgDistance = 0.0f;

            for (int i = 0, n = deformVerts.Length; i < n; ++i)
            {
                Vector3 vs = skinnedVerts[i], vd = deformVerts[i];

                var sqrDist = (vd - vs).sqrMagnitude;
                avgDistance += sqrDist;
                maxDistance = Math.Max(maxDistance, sqrDist);
            }

            avgDistance /= deformVerts.Length;

            DestroyImmediate(skinnedNeutralInstance);
            DestroyImmediate(deformNeutralInstance);

            Debug.Log($"Match error: mean {Mathf.Sqrt(avgDistance):0.####}; max {Mathf.Sqrt(maxDistance):0.####}");
            return true;
        }

        /// <summary>
        /// Try getting vertex mapping from UNIQUE skinned vertices to deformed vertices.
        /// </summary>
        /// <param name="neutralMesh">The skinned mesh in default/neutral pose.</param>
        /// <param name="deformSample">The path of file containing deformed outfit data.</param>
        /// <param name="uniqueIndices">The unique vertex indices of the skinned mesh.</param>
        /// <param name="vertexTable">The index mapping from true vertices to reduced ones.</param>
        /// <param name="mapping">The result vertex mapping table. The i-th element represents the index of vertex in deformed mesh corresponding to the i-th vertex in skinned mesh.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private bool TryGetUniqueVertexMapping(Mesh neutralMesh, string deformSample, out int[] uniqueIndices, out int[] vertexTable, out int[] mapping)
        {
            uniqueIndices = null;
            vertexTable = null;
            mapping = null;

            // Variables that will only be used when useAlembicForVertexMatching is true.
            Vector3[] skinnedNeutralVerts = null;
            Vector3[] deformNeutralVerts = null;
            KdTree3 skinnedNeutralTree = null;

            if (useAlembicForVertexMatching)
            {
                if (!TryMatchUniqueVerticesAtNeutralPose(out skinnedNeutralVerts, out deformNeutralVerts))
                {
                    return false;
                }
                skinnedNeutralTree = new KdTree3(skinnedNeutralVerts, skinnedNeutralVerts.Length);
            }

            if (!TryGetDeformInstance(deformSample, out var deformInstance, out var deformMesh, out var player))
            {
                return false;
            }
            player.UpdateImmediately(0.0f);

            GetUniqueVertices(neutralMesh, out uniqueIndices, out vertexTable);

            var skinnedVerts = neutralMesh.vertices;
            var deformVerts = deformMesh.vertices;
            KdTree3 deformTree = new KdTree3(deformVerts, deformVerts.Length);

            mapping = new int[uniqueIndices.Length];

            var maxDistance = 0.0f;
            var avgDistance = 0.0f;

            for (int i = 0, n = uniqueIndices.Length; i < n; ++i)
            {
                int deformIndex;
                var vs = skinnedVerts[uniqueIndices[i]];

                if (useAlembicForVertexMatching)
                {
                    var neutralIndex = skinnedNeutralTree.FindNearest(ref vs);

                    var vsn = skinnedNeutralVerts[neutralIndex];
                    var vdn = deformNeutralVerts[neutralIndex];

                    deformIndex = deformTree.FindNearest(ref vdn);
                }
                else
                {
                    deformIndex = deformTree.FindNearest(ref vs);
                }

                mapping[i] = deformIndex;

                var vd = deformVerts[deformIndex];
                var sqrDist = (vd - vs).sqrMagnitude;
                avgDistance += sqrDist;
                maxDistance = Math.Max(maxDistance, sqrDist);
            }

            avgDistance /= deformVerts.Length;

            Debug.Log($"Mapping error: mean {Mathf.Sqrt(avgDistance):0.####}; max {Mathf.Sqrt(maxDistance):0.####}");

            DestroyImmediate(deformInstance);

            HashSet<int> uniqueMapping = new HashSet<int>(mapping);
            if (uniqueMapping.Count != mapping.Length)
            {
                string warningMsg = "Some vertices in skinned mesh map to the same one in deform mesh. " +
                    "This may cause artifacts after deformation inference. ";

                if (useAlembicForVertexMatching)
                {
                    warningMsg += "Please check whether the Alembic files contain irrelevant information, such as normals and UVs.";
                }
                else
                {
                    warningMsg += "Please use alembic files to achieve better matching results.";
                }

                Debug.LogWarning(warningMsg);
            }

            return true;
        }

        /// <summary>
        /// Main entrance of the coroutine to process deformer data.
        /// </summary>
        /// <returns></returns>
        public IEnumerator ProcessDeformerData()
        {
            // Preparatory Work:
            // 1. Validate input parameters.
            // 2. Obtain vertex patch info of the outfit mesh.
            // 3. Obtain vertex mapping from skinned mesh to deformed mesh.

            // Be sure to disable deformer inference in this stage.
            var sentisDeformer = target.GetComponentInChildren<NeuralDeformerPlayer>();
            var sentisDeformerEnabled = sentisDeformer != null && sentisDeformer.enabled;
            if (sentisDeformerEnabled)
            {
                EditorUtility.DisplayDialog("", 
                    $"An active NeuralDeformerPlayer component is detected in target {target.name}. " +
                    $"It will be deactivated for accurately processing deform data.", 
                    "OK");
                sentisDeformer.enabled = false;
            }
            yield return null;

            EditorUtility.DisplayProgressBar("Deformer Data Processing", "Loading...", 0.0f);
            if (!TryValidate(out var neutralMesh, out var posesPaths, out var deformPaths))
            {
                EditorUtility.ClearProgressBar();
                yield break;
            }

            if (!TryGetUniqueVertexMapping(neutralMesh, deformPaths[0], out var uniqueIndices, out var vertexTable, out var mapping))
            {
                EditorUtility.ClearProgressBar();
                yield break;
            }

            // Extract a paired dataset where:
            // - Feature: Rotations of the character's key joints in each frame;
            // - Target:  Displacements/offsets of outfit mesh vertices.
            var smr = target.GetComponent<SkinnedMeshRenderer>();

            var ensuredExportDir = EnsureExportDirectory(Path.Combine(datasetDir, taskName), out var xSubdir, out var ySubdir, out var isInAssets);
            var minDelta = float.MaxValue;
            var maxDelta = float.MinValue;

            for (int p = 0, pn = posesPaths.Length; p < pn; ++p)
            {
                // For each animation, we get the skinned mesh and deformed mesh.
                if (!TryGetPoseClip(posesPaths[p], out var poseName, out var poseClip))
                {
                    EditorUtility.ClearProgressBar();
                    yield break;
                }

                if (!TryGetDeformInstance(deformPaths[p], out var deformInstance, out var deformMesh, out var player))
                {
                    EditorUtility.ClearProgressBar();
                    yield break;
                }

                float baseProgress = p / (float)pn;

                var rotations = new List<Quaternion>();
                var deltas = new List<Vector3>();

                GameObject pureDeformInstance = null;
                Mesh pureDeformMesh = null;
                //Mesh redeformMesh = null;
                //Mesh rebakedMesh = null;
                if (extractPureDeformation)
                {
                    string pureDeformInstanceName = $"{avatar.name}(PureDeform)";
                    string pureDeformMeshName = $"{target.name}(PureDeform)";

                    pureDeformInstance = GameObject.Find(pureDeformInstanceName);
                    if (pureDeformInstance == null)
                    {
                        pureDeformInstance = new GameObject(pureDeformInstanceName);
                        var mf = pureDeformInstance.AddComponent<MeshFilter>();
                        var mr = pureDeformInstance.AddComponent<MeshRenderer>();
                        pureDeformMesh = Instantiate(smr.sharedMesh);
                        pureDeformMesh.name = pureDeformMeshName;
                        mf.sharedMesh = pureDeformMesh;
                        mr.sharedMaterial = smr.sharedMaterial;
                        pureDeformInstance.transform.position = avatar.transform.position + new Vector3(-1.5f, 0, 0);
                    }
                    else
                    {
                        pureDeformMesh = pureDeformInstance.GetComponent<MeshFilter>().sharedMesh;
                    }
                }

                var during = Mathf.Min(poseClip.length, player.Duration);
                var frameCount = Mathf.RoundToInt(during * poseClip.frameRate);
                //var avgMSE = 0.0f;
                //var maxMSE = 0.0f;

                for (int f = 1; f < frameCount; ++f)
                {
                    float progress = baseProgress + ((f + 1) / (float)(frameCount * pn));
                    EditorUtility.DisplayProgressBar("Deformer Data Processing", $"Export {poseName} @ frame {f + 1}/{frameCount}", progress);

                    // For each frame, we sample joint rotations and vertex displacements.
                    var time = f / poseClip.frameRate;
                    poseClip.SampleAnimation(avatar, time);
                    player.UpdateImmediately(time);

                    rotations.AddRange(joints.ConvertAll(j => j.localRotation));

                    var neutralVerts = neutralMesh.vertices;
                    var deformVerts = deformMesh.vertices;

                    Vector3[] neutralDeformVerts = null;
                    if (extractPureDeformation)
                    {
                        neutralDeformVerts = ConvertSkinnedToNeutral(
                            smr,
                            neutralMesh,            // Neutral Mesh
                            deformMesh.vertices,    // Vertices in animated state
                            uniqueIndices,          // List of unique vertex indices to process
                            mapping
                        );
                        // Update the vertices of pureDeformInstance
                        if (pureDeformMesh != null && vertexTable != null)
                        {
                            Vector3[] fullVerts = new Vector3[vertexTable.Length];
                            for (int vi = 0; vi < vertexTable.Length; vi++)
                            {
                                int uniqueIdx = vertexTable[vi];
                                fullVerts[vi] = neutralDeformVerts[uniqueIdx];
                            }
                            pureDeformMesh.vertices = fullVerts;
                            pureDeformMesh.RecalculateNormals();
                        }
                    }
                    else
                    {
                        neutralDeformVerts = new Vector3[uniqueIndices.Length];
                        for (int i = 0; i < uniqueIndices.Length; ++i)
                        {
                            neutralDeformVerts[i] = deformVerts[mapping[i]];
                        }
                    }

                    Vector3[] perFrameDeltas = new Vector3[uniqueIndices.Length];
                    for (int i = 0; i < uniqueIndices.Length; ++i)
                    {
                        var delta = neutralDeformVerts[i] - neutralVerts[uniqueIndices[i]];
                        perFrameDeltas[i] = delta;

                        minDelta = Mathf.Min(minDelta, delta.x, delta.y, delta.z);
                        maxDelta = Mathf.Max(maxDelta, delta.x, delta.y, delta.z);
                    }
                    deltas.AddRange(perFrameDeltas);

                    //redeformMesh ??= Instantiate(smr.sharedMesh);
                    //var redeformVerts = neutralVerts.ToArray();
                    //for (int vi = 0; vi < vertexTable.Length; vi++)
                    //{
                    //    int uniqueIndex = vertexTable[vi];
                    //    redeformVerts[vi] += perFrameDeltas[uniqueIndex];
                    //}
                    //redeformMesh.vertices = redeformVerts;
                    //var originalMesh = smr.sharedMesh;
                    //smr.sharedMesh = redeformMesh;
                    //rebakedMesh ??= new Mesh();
                    //smr.BakeMesh(rebakedMesh);
                    //smr.sharedMesh = originalMesh;
                    //var bakedVerts = rebakedMesh.vertices;

                    //var perFrameMSE = 0.0f;

                    //for (int i = 0; i < uniqueIndices.Length; ++i)
                    //{
                    //    var v1 = bakedVerts[uniqueIndices[i]];
                    //    var v2 = deformVerts[mapping[i]];

                    //    var sqrDist = (v2 - v1).sqrMagnitude;
                    //    perFrameMSE += sqrDist;
                    //}

                    //perFrameMSE /= uniqueIndices.Length;

                    //maxMSE = Mathf.Max(maxMSE, perFrameMSE);
                    //avgMSE += perFrameMSE;

                    yield return null;
                }

                //avgMSE /= frameCount;
                //Debug.Log($"Cycle consistency error: mean {Mathf.Sqrt(avgMSE):0.####}; max {Mathf.Sqrt(maxMSE):0.####}");

                poseClip.SampleAnimation(avatar, 0.0f);
                DestroyImmediate(deformInstance);
                if (pureDeformInstance != null)
                {
                    DestroyImmediate(pureDeformInstance);
                }

                // Export features and targets to binary files.
                EditorUtility.DisplayProgressBar("Deformer Data Processing", $"Export {poseName}...", baseProgress + 1 / (float)(pn));
                if (!TryExportXAndY(p, rotations, deltas, xSubdir, ySubdir, isInAssets))
                {
                    EditorUtility.ClearProgressBar();
                    yield break;
                }

                yield return null;
            }

            // Export patch and meta info.
            EditorUtility.DisplayProgressBar("Deformer Data Processing", "Export MetaInfo...", 1.0f);
            if (!TryExportMetaInfo(vertexTable, uniqueIndices, minDelta, maxDelta, ensuredExportDir, isInAssets))
            {
                EditorUtility.ClearProgressBar();
                yield break;
            }

            if (isInAssets)
            {
                AssetDatabase.Refresh();
            }

            EditorUtility.ClearProgressBar();
        }

        /// <summary>
        /// Utilizes SkinnedMeshRenderer.BakeMesh to perform perturbation sampling on the neutral model.
        /// The goal is to construct a precise, skinning-affected local coordinate system for each vertex in the current animation frame.
        /// This local system is then used to transform a vertex that includes both "animation + deformation" back into a "neutral space" that only contains the "deformation".
        /// Why perform this conversion? In machine learning, we want the model to learn only the "secondary deformations" (e.g., cloth wrinkles, muscle compression),
        /// not the skeletal animation itself. This function mathematically removes the influence of the skeletal animation,
        /// ensuring that the final calculated vertex `delta` is a clean training target representing only this secondary deformation.
        /// Note: If extractPureDeformation is false, the input mesh (e.g., from Alembic) must already be in the deformed neutral pose,
        /// i.e., it should contain only the secondary deformation and no skeletal animation.
        /// In this case, no animation decoupling is performed, and the function expects the input to be pre-aligned to the neutral space.
        /// </summary>
        /// <param name="smr">The target SkinnedMeshRenderer used for the BakeMesh operation.</param>
        /// <param name="neutralMesh">The baseline Mesh in its neutral state (without any deformation).</param>
        /// <param name="deformVerts">The vertex array from the current animation frame, containing both "animation" and "deformation".</param>
        /// <param name="uniqueIndices">A list of unique vertex indices to process, for optimization.</param>
        /// <param name="mapping">An index mapping from uniqueIndices to the deformVerts array.</param>
        /// <returns>A vertex array corresponding to `uniqueIndices`, transformed back into the "neutral space".</returns>
        private Vector3[] ConvertSkinnedToNeutral(
            SkinnedMeshRenderer smr,
            Mesh neutralMesh,
            Vector3[] deformVerts,
            int[] uniqueIndices,
            int[] mapping)
        {
            // --- Step 1: Prepare Perturbation Data ---
            // Clone the neutral mesh to avoid modifying the original asset. This is the basis for all operations.
            Mesh cloneNeutral = Instantiate(neutralMesh);
            Vector3[] verts0 = cloneNeutral.vertices;

            // Construct three new meshes, each with a small displacement (epsilon) applied along the X, Y, and Z axes.
            // These perturbed meshes will be used to sample tangent vectors via BakeMesh.
            Mesh meshX = Instantiate(cloneNeutral);
            Mesh meshY = Instantiate(cloneNeutral);
            Mesh meshZ = Instantiate(cloneNeutral);

            var vx = verts0.Select(v => v + Vector3.right * k_PerturbationEpsilon).ToArray();
            var vy = verts0.Select(v => v + Vector3.up * k_PerturbationEpsilon).ToArray();
            var vz = verts0.Select(v => v + Vector3.forward * k_PerturbationEpsilon).ToArray();

            meshX.vertices = vx;
            meshY.vertices = vy;
            meshZ.vertices = vz;

            // --- Step 2: Sample with BakeMesh ---
            // BakeMesh captures the skinning result of the input mesh in the current animation pose.
            // We bake the neutral mesh and the three perturbed meshes to get their true skinned shapes in the current frame.
            Mesh baked0 = new Mesh();
            Mesh bakedX = new Mesh();
            Mesh bakedY = new Mesh();
            Mesh bakedZ = new Mesh();

            var originalMesh = smr.sharedMesh;
            smr.sharedMesh = cloneNeutral; smr.BakeMesh(baked0);
            smr.sharedMesh = meshX;       smr.BakeMesh(bakedX);
            smr.sharedMesh = meshY;       smr.BakeMesh(bakedY);
            smr.sharedMesh = meshZ;       smr.BakeMesh(bakedZ);
            smr.sharedMesh = originalMesh;

            // --- Step 3: Calculate the True Local Coordinate System ---
            // Get the vertices of the baked meshes. These now include the influence of skeletal animation.
            var baseVerts = baked0.vertices;
            var bx = bakedX.vertices;
            var by = bakedY.vertices;
            var bz = bakedZ.vertices;

            // --- Step 4: Perform Local Space Inverse Transformation for Each Vertex ---
            Vector3[] result = new Vector3[uniqueIndices.Length];
            for (int i = 0; i < uniqueIndices.Length; i++)
            {
                var idx = uniqueIndices[i];
                // p0 is the position of the neutral mesh vertex after being skinned to the current pose. It serves as the origin of our local coordinate system.
                Vector3 p0 = baseVerts[idx];
                // By taking the difference between the baked perturbed vertices and the baked origin, we get the true local coordinate axes (tangent vectors) that have been "warped" by the skinning.
                // Dividing by epsilon normalizes the difference, giving us the rate of change for a unit perturbation.
                Vector3 ex = (bx[idx] - p0) / k_PerturbationEpsilon;
                Vector3 ey = (by[idx] - p0) / k_PerturbationEpsilon;
                Vector3 ez = (bz[idx] - p0) / k_PerturbationEpsilon;

                // Construct a transformation matrix from this "true local space" to world space.
                // This is equivalent to the extractMatrix in Maya's extractDeltas algorithm.
                Matrix4x4 M = new Matrix4x4();
                M.SetColumn(0, new Vector4(ex.x, ex.y, ex.z, 0));
                M.SetColumn(1, new Vector4(ey.x, ey.y, ey.z, 0));
                M.SetColumn(2, new Vector4(ez.x, ez.y, ez.z, 0));
                M.SetColumn(3, new Vector4(p0.x, p0.y, p0.z, 1));

                // Transform the simulated vertex (containing "animation + deformation") back into the local space using the inverse of the matrix.
                // This step effectively "subtracts" the influence of the skeletal animation.
                Vector3 simulated = deformVerts[mapping[i]];
                Vector3 local     = M.inverse.MultiplyPoint3x4(simulated);
                
                // Finally, add this "pure deformation" local offset to the original neutral vertex to get the final result.
                result[i] = verts0[idx] + local;            
            }

            // --- Step 5: Clean Up Temporary Objects ---
            DestroyImmediate(cloneNeutral);
            DestroyImmediate(meshX);
            DestroyImmediate(meshY);
            DestroyImmediate(meshZ);
            
            return result;
        }
        #endregion

        #region Output
        /// <summary>
        /// Ensure the export directory and sub-direcotries exist.
        /// </summary>
        /// <param name="exportDir">The export directory.</param>
        /// <param name="xSubdir">The result sub-directory where features are saved.</param>
        /// <param name="ySubdir">The result sub-directory where targets are saved.</param>
        /// <param name="isInAssets">The result flag indicates whether <paramref name="exportDir"/> is in current project's "Assets/" directory.</param>
        /// <returns>(<see cref="string"/>) The absolute/full path of the ensured export directory.</returns>
        private static string EnsureExportDirectory(string exportDir, out string xSubdir, out string ySubdir, out bool isInAssets)
        {
            var fullExportDir = Path.GetFullPath(exportDir);
            var fullAssetsDir = Path.GetFullPath(Application.dataPath);

            if (!Directory.Exists(fullExportDir))
            {
                Directory.CreateDirectory(fullExportDir);
            }

            xSubdir = Path.Combine(fullExportDir, "x");
            if (!Directory.Exists(xSubdir))
            {
                Directory.CreateDirectory(xSubdir);
            }

            ySubdir = Path.Combine(fullExportDir, "y");
            if (!Directory.Exists(ySubdir))
            {
                Directory.CreateDirectory(ySubdir);
            }

            isInAssets = fullExportDir.StartsWith(fullAssetsDir);

            if (isInAssets)
            {
                AssetDatabase.Refresh();
            }

            return fullExportDir;
        }

        /// <summary>
        /// Try exporting features and targets to binary files.
        /// </summary>
        /// <param name="index">The animation index.</param>
        /// <param name="rotations">The list of joint rotation quaternions.</param>
        /// <param name="deltas">The list of mesh vertex displacements.</param>
        /// <param name="xSubdir">The feature sub-directory.</param>
        /// <param name="ySubdir">The target sub-directory.</param>
        /// <param name="isInAssets">The result flag indicates whether <paramref name="xSubdir"/> and <paramref name="ySubdir"/> are in current project's "Assets/" directory.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private bool TryExportXAndY(int index, List<Quaternion> rotations, List<Vector3> deltas, string xSubdir, string ySubdir, bool isInAssets)
        {
            try
            {
                using (var xWriter = new BinaryWriter(File.Open(Path.Combine(xSubdir, String.Format("pose_{0:0000}.bin", index)), FileMode.Create)))
                {
                    foreach (var rotation in rotations)
                    {
                        xWriter.Write(rotation.x);
                        xWriter.Write(rotation.y);
                        xWriter.Write(rotation.z);
                        xWriter.Write(rotation.w);
                    }
                    xWriter.Flush();
                }

                using (var yWriter = new BinaryWriter(File.Open(Path.Combine(ySubdir, String.Format("pose_{0:0000}.bin", index)), FileMode.Create)))
                {
                    foreach (var delta in deltas)
                    {
                        yWriter.Write(delta.x);
                        yWriter.Write(delta.y);
                        yWriter.Write(delta.z);
                    }
                    yWriter.Flush();
                }
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Export", e.Message, "OK");
                return false;
            }

            return true;
        }

        /// <summary>
        /// Try exporting patch and meta info.
        /// </summary>
        /// <param name="vertexTable">The index mapping from true vertices to reduced ones.</param>
        /// <param name="uniqueIndices">The unique vertex indices of the skinned mesh.</param>
        /// <param name="deltaMin">The minimum vertex displacement.</param>
        /// <param name="deltaMax">The maximum vertex displacement.</param>
        /// <param name="fullExportDir">The full path of the export directory.</param>
        /// <param name="isInAssets">The result flag indicates whether <paramref name="fullExportDir"/> is in current project's "Assets/" directory.</param>
        /// <returns>(<see cref="bool"/>) If it is done without errors.</returns>
        private bool TryExportMetaInfo(int[] vertexTable, int[] uniqueIndices, float deltaMin, float deltaMax, string fullExportDir, bool isInAssets)
        {
            try
            {
                using (var metaWriter = new StreamWriter(File.Open(Path.Combine(fullExportDir, "meta.txt"), FileMode.Create)))
                {
                    metaWriter.WriteLine($"JointCount:{joints.Count}");
                    metaWriter.WriteLine($"VertexCount:{vertexTable.Length}");
                    metaWriter.WriteLine($"UniqueVertexCount:{uniqueIndices.Length}");
                    metaWriter.WriteLine($"DeltaMin:{deltaMin}");
                    metaWriter.WriteLine($"DeltaMax:{deltaMax}");
                    metaWriter.Flush();
                }

                var metaAsset = ScriptableObject.CreateInstance<NeuralDeformerDatasetMetaInfo>();
                metaAsset.Init(joints, vertexTable, uniqueIndices.Length, deltaMin, deltaMax);
                var metaPath = Path.Combine(isInAssets ? Path.Combine(datasetDir, taskName) : "Assets", "NeuralDeformerDatasetMetaInfo.asset");
                AssetDatabase.CreateAsset(metaAsset, metaPath);
                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();
            }
            catch (Exception e)
            {
                EditorUtility.DisplayDialog("Export", e.Message, "OK");
                return false;
            }

            return true;
        }

        #endregion
#endif
    }
}