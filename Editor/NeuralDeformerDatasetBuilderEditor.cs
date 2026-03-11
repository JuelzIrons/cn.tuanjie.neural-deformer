using System.Collections;
using System.Collections.Generic;
using Unity.EditorCoroutines.Editor;
using UnityEngine;
using UnityEditor;
using UnityEditor.UIElements;
using UnityEngine.UIElements;
using System.IO;
using System.Linq;
using UnityEngine.Assertions;
using System;

namespace Tuanjie.NeuralDeformer
{
    /// <summary>
    /// <see cref="NeuralDeformerDatasetBuilderEditor"> class implements Inspector UI for <see cref="NeuralDeformerDatasetBuilder">.
    /// </summary>
    [CustomEditor(typeof(NeuralDeformerDatasetBuilder))]
    internal class NeuralDeformerDatasetBuilderEditor : UnityEditor.Editor
    {
        private NeuralDeformerDatasetBuilder tool => (NeuralDeformerDatasetBuilder)target;

        private SerializedProperty m_Avatar;
        private SerializedProperty m_Target;
        private SerializedProperty m_RootBone;
        private SerializedProperty m_Joints;
        private SerializedProperty m_UseAlembic;
        private SerializedProperty m_NeutralSkinnedAlembic;
        private SerializedProperty m_NeutralDeformAlembic;
        private SerializedProperty m_extractPureDeformation;
        private SerializedProperty m_PosesDir;
        private SerializedProperty m_DeformDir;
        private SerializedProperty m_DatasetDir;

        private Button m_LoadDefaultButton;
        private Button m_ExportButton;
        private Button m_RevealButton;
        private const string k_ButtonRowClass = "button-row";


        void OnEnable()
        {
            m_Avatar = serializedObject.FindProperty(nameof(tool.avatar));
            m_Target = serializedObject.FindProperty(nameof(tool.target));
            m_RootBone = serializedObject.FindProperty(nameof(tool.rootBone));
            m_Joints = serializedObject.FindProperty(nameof(tool.joints));
            m_UseAlembic = serializedObject.FindProperty(nameof(tool.useAlembicForVertexMatching));
            m_NeutralSkinnedAlembic = serializedObject.FindProperty(nameof(tool.skinnedAlembic));
            m_NeutralDeformAlembic = serializedObject.FindProperty(nameof(tool.deformAlembic));
            m_extractPureDeformation = serializedObject.FindProperty(nameof(tool.extractPureDeformation));
            m_PosesDir = serializedObject.FindProperty(nameof(tool.posesDir));
            m_DeformDir = serializedObject.FindProperty(nameof(tool.deformDir));
            m_DatasetDir = serializedObject.FindProperty(nameof(tool.datasetDir));
        }

        public override VisualElement CreateInspectorGUI()
        {
            var root = new VisualElement();
            root.styleSheets.Add(Resources.Load<StyleSheet>("ButtonRow"));
            root.styleSheets.Add(Resources.Load<StyleSheet>("PreprocessEditor"));

            root.Add(new PropertyField(m_Avatar));

            var targetField = new PropertyField(m_Target, "Target Of Deformation");
            targetField.RegisterValueChangeCallback(evt =>
            {
                Undo.RecordObject(tool, "Auto Switch Root Bone");
                if (tool.target == null || tool.target.GetComponentInChildren<SkinnedMeshRenderer>() == null)
                {
                    tool.rootBone = null;
                    tool.joints.Clear();
                    tool.taskName = null;
                }
                else
                {
                    // Ensure that the root bone is set to the first bone of the target's skeleton.
                    var smr = tool.target.GetComponentInChildren<SkinnedMeshRenderer>();
                    tool.rootBone = smr.rootBone;
                    tool.taskName = ObjectNames.NicifyVariableName(tool.target.name).Replace(" ", "_") ?? DateTime.Now.ToString("yyMMdd-HHmmss");
                }
                EditorUtility.SetDirty(tool);
            });
            root.Add(targetField);

            var rootBoneField = new VisualElement();
            rootBoneField.style.flexDirection = FlexDirection.Row;
            rootBoneField.Add(new PropertyField(m_RootBone) { style = { flexGrow = 1, paddingRight = 10 } });
            var jointsSelectionButton = new Button(() =>
            {
                if (tool.rootBone == null)
                {
                    EditorUtility.DisplayDialog("Error", "Root Bone is not set. Please set it first.", "OK");
                }
                else
                {
                    var popupWindow = JointsSelectionPopupWindow.Open(tool.rootBone, tool.joints);
                    popupWindow.onJointsSelected += joints =>
                    {
                        Undo.RecordObject(tool, "Set Joints");
                        tool.joints = joints;
                        EditorUtility.SetDirty(tool);
                    };
                }
            })
            {
                text = "Select Joints...",
                tooltip = "",
            };
            rootBoneField.Add(jointsSelectionButton);
            root.Add(rootBoneField);

            root.Add(new PropertyField(m_Joints));

            var useAlembicField = new PropertyField(m_UseAlembic);
            var neutralSkinnedAlembicField = new PropertyField(m_NeutralSkinnedAlembic);
            var neutralDeformAlembicField = new PropertyField(m_NeutralDeformAlembic);
            useAlembicField.RegisterValueChangeCallback(evt =>
            {
                neutralSkinnedAlembicField.style.display = evt.changedProperty.boolValue ? DisplayStyle.Flex : DisplayStyle.None;
                neutralDeformAlembicField.style.display = evt.changedProperty.boolValue ? DisplayStyle.Flex : DisplayStyle.None;
            });
            root.Add(useAlembicField);
            root.Add(neutralSkinnedAlembicField);
            root.Add(neutralDeformAlembicField);
            root.Add(new PropertyField(m_extractPureDeformation));
            root.Add(new PropertyField(m_PosesDir));
            root.Add(new PropertyField(m_DeformDir));
            root.Add(new PropertyField(m_DatasetDir));

            var buttonRow = new VisualElement();
            buttonRow.AddToClassList(k_ButtonRowClass);
            root.Add(buttonRow);

            // Export Dataset Button
            m_ExportButton = new Button() { text = "Export" };
            m_ExportButton.RegisterCallback<ClickEvent>(evt =>
            {
                EditorCoroutineUtility.StartCoroutine(ExportCoroutine(), tool);
            });
            buttonRow.Add(m_ExportButton);

            // Reveal Dataset Button
            m_RevealButton = new Button() { text = "Reveal Dataset" };
            m_RevealButton.RegisterCallback<ClickEvent>(evt =>
            {
                if (string.IsNullOrEmpty(tool.datasetDir))
                {
                    EditorUtility.DisplayDialog("Error", "Dataset Dir is empty. Please set it with valid path.", "OK");
                }
                else if (string.IsNullOrEmpty(tool.taskName))
                {
                    EditorUtility.DisplayDialog("Error", "Task Name is empty. Please set it first.", "OK");
                }
                else
                {
                    string metaFile = Path.Combine(tool.datasetDir, tool.taskName, "meta.txt");
                    if (!File.Exists(metaFile))
                    {
                        EditorUtility.DisplayDialog("Error", "Dataset directory does not exist. Please export first.", "OK");
                    }
                    else
                    {
                        var fullMetaPath = Path.GetFullPath(metaFile);
                        var fullAssetsPath = Path.GetFullPath(Application.dataPath);

                        if (fullMetaPath.StartsWith(fullAssetsPath))
                        {
                            var revealPath = fullMetaPath.Replace(fullAssetsPath, "Assets");
                            var lossTex = AssetDatabase.LoadAssetAtPath<TextAsset>(revealPath);
                            EditorGUIUtility.PingObject(lossTex);
                            Selection.activeObject = lossTex;
                            EditorUtility.FocusProjectWindow();
                        }
                        else
                        {
                            EditorUtility.RevealInFinder(fullMetaPath);
                        }
                    }
                }
            });
            buttonRow.Add(m_RevealButton);

            return root;
        }

        /// <summary>
        /// Export dataset, which will be used for deformer training.
        /// </summary>
        /// <returns></returns>
        private IEnumerator ExportCoroutine()
        {
            m_LoadDefaultButton?.SetEnabled(false);
            m_ExportButton.SetEnabled(false);
            m_RevealButton.SetEnabled(false);

            // Be sure to disable deformer inference in this stage.
            var sentisDeformer = tool.gameObject.GetComponentInChildren<NeuralDeformerPlayer>();
            var sentisDeformerEnabled = sentisDeformer != null && sentisDeformer.enabled;
            if (sentisDeformerEnabled)
            {
                sentisDeformer.enabled = false;
            }

            yield return tool.ProcessDeformerData();

            if (sentisDeformerEnabled)
            {
                sentisDeformer.enabled = true;
            }

            m_LoadDefaultButton?.SetEnabled(true);
            m_ExportButton.SetEnabled(true);
            m_RevealButton.SetEnabled(true);
        }
    }

}