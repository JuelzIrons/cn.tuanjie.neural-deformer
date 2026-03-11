using UnityEditor;
using UnityEngine.UIElements;
using UnityEditor.UIElements;
using System.IO;
using Unity.Sentis;

namespace Tuanjie.NeuralDeformer
{
    [CustomEditor(typeof(NeuralDeformerPlayer))]
    internal class NeuralDeformerPlayerEditor : UnityEditor.Editor
    {
        private NeuralDeformerPlayer tool => (NeuralDeformerPlayer)target;

        private SerializedProperty m_EnableDeformation;
        private SerializedProperty m_DeformerDataMetaInfo;
        private SerializedProperty m_ModelAsset;
        private SerializedProperty m_InferenceBackend;
        private SerializedProperty m_DeformBackend;
        private SerializedProperty m_DeformationWeight;
        private SerializedProperty m_AlphaMaskInfluenceWeight;
        private SerializedProperty m_RecalculateNormals;
        private SerializedProperty m_EnableDeformationDebuging;
        private SerializedProperty m_EnableDeformationLogging;

        void OnEnable()
        {
            m_EnableDeformation = serializedObject.FindProperty(nameof(tool.enableDeformation));
            m_DeformerDataMetaInfo = serializedObject.FindProperty(nameof(tool.neuralDeformerDatasetMetaInfo));
            m_ModelAsset = serializedObject.FindProperty(nameof(tool.modelAsset));
            m_InferenceBackend = serializedObject.FindProperty("_inferenceBackend");
            m_DeformBackend = serializedObject.FindProperty("_deformBackend");
            m_DeformationWeight = serializedObject.FindProperty(nameof(tool.deformationWeight));
            m_AlphaMaskInfluenceWeight = serializedObject.FindProperty(nameof(tool.alphaMaskInfluenceWeight));
            m_RecalculateNormals = serializedObject.FindProperty(nameof(tool.recalculateNormals));
            m_EnableDeformationDebuging = serializedObject.FindProperty(nameof(tool.enableDeformationDebuging));
            m_EnableDeformationLogging = serializedObject.FindProperty(nameof(tool.enableDeformationLogging));
        }

        public override VisualElement CreateInspectorGUI()
        {
            var root = new VisualElement();

            var enableDeformationField = new PropertyField(m_EnableDeformation);
            enableDeformationField.RegisterValueChangeCallback(evt =>
            {
                tool.NotifyResetDataResources();
            });
            root.Add(enableDeformationField);

            var deformerDataMetaInfoField = new PropertyField(m_DeformerDataMetaInfo);
            deformerDataMetaInfoField.RegisterValueChangeCallback(evt =>
            {
                tool.NotifyResetDataResources();
            });
            root.Add(deformerDataMetaInfoField);

            var modelAssetField = new PropertyField(m_ModelAsset);
            var switchBtn = new Button(SwitchToSentis) { text = "Switch to Sentis Format (Recommended)" };
            modelAssetField.RegisterValueChangeCallback(evt =>
            {
                var obj = evt.changedProperty.objectReferenceValue;
                bool isONNXFormat = obj != null && AssetDatabase.GetAssetPath(obj).EndsWith(".onnx");
                switchBtn.style.display = isONNXFormat ? DisplayStyle.Flex : DisplayStyle.None;

                tool.NotifyResetDataResources();
            });
            root.Add(modelAssetField);
            root.Add(switchBtn);

            var sentisBackendField = new PropertyField(m_InferenceBackend);
            sentisBackendField.RegisterValueChangeCallback(evt =>
            {
                tool.NotifyResetDataResources();
            });
            root.Add(sentisBackendField);

            var deformBackendField = new PropertyField(m_DeformBackend);
            deformBackendField.RegisterValueChangeCallback(evt =>
            {
                tool.NotifyResetDataResources();
            });
            root.Add(deformBackendField);

            root.Add(new PropertyField(m_DeformationWeight));
            root.Add(new PropertyField(m_AlphaMaskInfluenceWeight));

            var recalculateNormalsField = new PropertyField(m_RecalculateNormals);
            recalculateNormalsField.RegisterValueChangeCallback(evt =>
            {
                tool.NotifyResetDataResources();
            });
            root.Add(recalculateNormalsField);

            var enableDebuggingField = new PropertyField(m_EnableDeformationDebuging, "Enable Debugging & Profiling");
            var enableLoggingField = new PropertyField(m_EnableDeformationLogging, "Enable Logging");

            enableDebuggingField.RegisterValueChangeCallback(evt =>
            {
                bool debugEnabled = m_EnableDeformationDebuging.boolValue;
                enableLoggingField.SetEnabled(debugEnabled);

                if (!debugEnabled && m_EnableDeformationLogging.boolValue)
                {
                    m_EnableDeformationLogging.boolValue = false;
                    serializedObject.ApplyModifiedProperties();
                }
            });

            enableLoggingField.SetEnabled(m_EnableDeformationDebuging.boolValue);

            root.Add(enableDebuggingField);
            root.Add(enableLoggingField);
            return root;
        }

        private void SwitchToSentis()
        {
            var modelAsset = m_ModelAsset.objectReferenceValue as ModelAsset;
            if (modelAsset != null)
            {
                string directory = Path.GetDirectoryName(AssetDatabase.GetAssetPath(modelAsset));
                string sentisPath = Path.Combine(directory, $"{modelAsset.name}.sentis");
                
                if (!File.Exists(sentisPath))
                {
                    ModelWriter.Save(sentisPath, modelAsset);
                    AssetDatabase.Refresh();
                }

                var sentisModelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(sentisPath);
                tool.modelAsset = sentisModelAsset;
            }
        }
    }

}