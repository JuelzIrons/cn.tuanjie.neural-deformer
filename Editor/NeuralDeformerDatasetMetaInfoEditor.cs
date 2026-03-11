using UnityEngine;
using UnityEditor;
using UnityEngine.UIElements;
using UnityEditor.UIElements;

namespace Tuanjie.NeuralDeformer
{
    [CustomEditor(typeof(NeuralDeformerDatasetMetaInfo))]
    internal class NeuralDeformerDatasetMetaInfoEditor : UnityEditor.Editor
    {
        NeuralDeformerDatasetMetaInfo tool => target as NeuralDeformerDatasetMetaInfo;

        SerializedProperty m_JointCount;
        SerializedProperty m_JointNames;
        SerializedProperty m_VertexCount;
        SerializedProperty m_UniqueVertexCount;
        SerializedProperty m_MinDelta;
        SerializedProperty m_MaxDelta;

        private void OnEnable()
        {
            m_JointCount = serializedObject.FindProperty(nameof(m_JointCount));
            m_JointNames = serializedObject.FindProperty(nameof(m_JointNames));
            m_VertexCount = serializedObject.FindProperty(nameof(m_VertexCount));
            m_UniqueVertexCount = serializedObject.FindProperty(nameof(m_UniqueVertexCount));
            m_MinDelta = serializedObject.FindProperty(nameof(m_MinDelta));
            m_MaxDelta = serializedObject.FindProperty(nameof(m_MaxDelta));
        }

        public override VisualElement CreateInspectorGUI()
        {
            var root = new VisualElement();
            root.Add(new PropertyField(m_JointCount));
            root.Add(new PropertyField(m_JointNames));
            root.Add(new PropertyField(m_VertexCount));
            root.Add(new PropertyField(m_UniqueVertexCount));
            root.Add(new PropertyField(m_MinDelta));
            root.Add(new PropertyField(m_MaxDelta));
            root.SetEnabled(false);
            return root;
        }
    }
}