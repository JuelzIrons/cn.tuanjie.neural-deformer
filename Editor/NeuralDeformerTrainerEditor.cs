using System.Collections;
using System;
using System.IO;
using Unity.EditorCoroutines.Editor;
using Unity.Sentis;
using UnityEngine;
using UnityEditor;
using UnityEditor.UIElements;
using UnityEngine.UIElements;

namespace Tuanjie.NeuralDeformer
{
    /// <summary>
    /// <see cref="NeuralDeformerTrainerEditor"/> class implements Inspector UI for <see cref="NeuralDeformerTrainer"/>.
    /// </summary>
    [CustomEditor(typeof(NeuralDeformerTrainer))]
    internal class NeuralDeformerTrainerEditor : UnityEditor.Editor
    {
        private NeuralDeformerTrainer tool => (NeuralDeformerTrainer)target;

        private SerializedProperty m_UseDefaultRunName;
        private SerializedProperty m_RunName;
        private SerializedProperty m_Seed;
        private SerializedProperty m_CentersNumber;
        private SerializedProperty m_CentersInit;
        private SerializedProperty m_BetasInit;
        private SerializedProperty m_BatchSize;
        private SerializedProperty m_Shuffle;
        private SerializedProperty m_Epochs;
        private SerializedProperty m_LearningRate;
        private SerializedProperty m_TaskName;
        private SerializedProperty m_DatasetDir;
        private SerializedProperty m_ReportDir;
        private SerializedProperty m_ModelsDir;
        private SerializedProperty m_EnableReport;
        private SerializedProperty m_Verbose;

        private Button m_LoadDefaultButton;
        private Button m_ResetParamButton;
        private Button m_TrainButton;
        private Button m_TerminateButton;
        private Button m_ClearVenvButton;

        private ProgressBar m_Progress;
        private VisualElement m_LossElement;
        private Image m_LossFigure;
        private TextField m_ModelNameField;
        private Button m_RevealLossButton;
        private Button m_RevealModelButton;
        private Button m_ClearResultButton;
        private const string k_ButtonRowClass = "button-row";
        private const string k_LossLoadedStyle = "loss-loaded";

        private void OnEnable()
        {
            m_UseDefaultRunName = serializedObject.FindProperty(nameof(tool.useDefaultRunName));
            m_RunName = serializedObject.FindProperty(nameof(tool.runName));
            m_Seed = serializedObject.FindProperty(nameof(tool.seed));
            m_CentersNumber = serializedObject.FindProperty(nameof(tool.centersNumber));
            m_CentersInit = serializedObject.FindProperty(nameof(tool.centersInit));
            m_BetasInit = serializedObject.FindProperty(nameof(tool.betasInit));
            m_BatchSize = serializedObject.FindProperty(nameof(tool.batchSize));
            m_Shuffle = serializedObject.FindProperty(nameof(tool.shuffle));
            m_Epochs = serializedObject.FindProperty(nameof(tool.epochs));
            m_LearningRate = serializedObject.FindProperty(nameof(tool.learningRate));
            m_TaskName = serializedObject.FindProperty(nameof(tool.taskName));
            m_DatasetDir = serializedObject.FindProperty(nameof(tool.datasetDir));
            m_ReportDir = serializedObject.FindProperty(nameof(tool.reportDir));
            m_ModelsDir = serializedObject.FindProperty(nameof(tool.modelsDir));
            m_EnableReport = serializedObject.FindProperty(nameof(tool.enableReport));
            m_Verbose = serializedObject.FindProperty(nameof(tool.verbose));

            tool.progressInfo.onProgressUpdated -= RenderProgressBar;
            tool.progressInfo.onProgressUpdated += RenderProgressBar;
        }

        private void OnDisable()
        {
            tool.progressInfo.onProgressUpdated -= RenderProgressBar;
        }

        /// <summary>
        /// Updates the state of the Clear Python .venv button based on the existence of the .venv directory.
        /// </summary>
        private void UpdateClearVenvButtonState()
        {
            if (m_ClearVenvButton != null)
            {
                m_ClearVenvButton.SetEnabled(tool.venvExists);
            }
        }

        public override VisualElement CreateInspectorGUI()
        {
            VisualElement root = new();
            root.styleSheets.Add(Resources.Load<StyleSheet>("ButtonRow"));
            root.styleSheets.Add(Resources.Load<StyleSheet>("LossElement"));

            VisualElement hyperParamsDiv = new();
            VisualElement dataSettingsDiv = new();

            var useDefaultRunNameField = new PropertyField(m_UseDefaultRunName);
            var runNameField = new PropertyField(m_RunName, "Run Name");
            useDefaultRunNameField.RegisterValueChangeCallback(evt =>
            {
                runNameField.style.display = evt.changedProperty.boolValue ? DisplayStyle.None : DisplayStyle.Flex;
            });
            hyperParamsDiv.Add(useDefaultRunNameField);
            hyperParamsDiv.Add(runNameField);

            var seedField = new PropertyField(m_Seed);
            hyperParamsDiv.Add(seedField);
            
            var centersNumberField = new PropertyField(m_CentersNumber);
            var centersInitField = new PropertyField(m_CentersInit);
            var betasInitField = new PropertyField(m_BetasInit);
            hyperParamsDiv.Add(centersNumberField);
            hyperParamsDiv.Add(centersInitField);
            hyperParamsDiv.Add(betasInitField);

            var batchSizeField = new PropertyField(m_BatchSize);
            var shuffleField = new PropertyField(m_Shuffle);
            var epochsField = new PropertyField(m_Epochs);
            var learningRateField = new PropertyField(m_LearningRate);
            hyperParamsDiv.Add(batchSizeField);
            hyperParamsDiv.Add(shuffleField);
            hyperParamsDiv.Add(epochsField);
            hyperParamsDiv.Add(learningRateField);

            m_ResetParamButton = new Button() { text = "Reset Hyper-parameter" };
            m_ResetParamButton.RegisterCallback<ClickEvent>(evt =>
            {
                EditorCoroutineUtility.StartCoroutine(ResetParameterCoroutine(), tool);
            });
            hyperParamsDiv.Add(m_ResetParamButton);

            var taskNameField = new PropertyField(m_TaskName);
            taskNameField.SetEnabled(false);
            var datasetDirField = new PropertyField(m_DatasetDir);
            datasetDirField.SetEnabled(false);
            var modelsDirField = new PropertyField(m_ModelsDir);
            var reportDirField = new PropertyField(m_ReportDir, "Logs Dir");
            dataSettingsDiv.Add(taskNameField);
            dataSettingsDiv.Add(datasetDirField);
            dataSettingsDiv.Add(modelsDirField);
            dataSettingsDiv.Add(reportDirField);

            m_LoadDefaultButton = new Button() { text = "Sync Dataset Settings" };
            m_LoadDefaultButton.RegisterCallback<ClickEvent>(evt =>
            {
                EditorCoroutineUtility.StartCoroutine(SyncDatasetCoroutine(), tool);
            });
            dataSettingsDiv.Add(m_LoadDefaultButton);

            root.Add(dataSettingsDiv);
            root.Add(hyperParamsDiv);

            var enableReportField = new PropertyField(m_EnableReport, "Enable Tensorboard Logs");
            root.Add(enableReportField);

            var verboseField = new PropertyField(m_Verbose);
            root.Add(verboseField);

            var mainButtonRow = new VisualElement();
            mainButtonRow.AddToClassList(k_ButtonRowClass);
            root.Add(mainButtonRow);

            m_TrainButton = new Button(OnClickTrainButton){ text = "Train" };
            m_TerminateButton = new Button(OnClickTerminateButton) { text = "Terminate" };
            m_TerminateButton.SetEnabled(false);
            m_ClearVenvButton = new Button() { text = "Clear Python .venv" };
            m_ClearVenvButton.RegisterCallback<ClickEvent>(evt =>
            {
                EditorCoroutineUtility.StartCoroutine(ClearVenvCoroutine(), tool);
            });
            UpdateClearVenvButtonState();
            mainButtonRow.Add(m_TrainButton);
            mainButtonRow.Add(m_TerminateButton);
            mainButtonRow.Add(m_ClearVenvButton);

            m_Progress = new ProgressBar()
            {
                lowValue = 0f,
                highValue = 1f,
                value = 0f
            };
            RenderProgressBar();
            root.Add(m_Progress);

            InitTrainingResults(root);

            return root;
        }

        #region Main Button Row

        /// <summary>
        /// Set default settings for the NeuralDeformerTrainer.
        /// </summary>
        /// <returns></returns>
        private IEnumerator SyncDatasetCoroutine()
        {
            tool.SyncDataset();
            yield return null;
        }

        /// <summary>
        /// Set default settings for the NeuralDeformerTrainer.
        /// </summary>
        /// <returns></returns>
        private IEnumerator ResetParameterCoroutine()
        {
            tool.ResetParameter();
            yield return null;
        }

        /// <summary>
        /// Train the model from Editor.
        /// </summary>
        private async void OnClickTrainButton()
        {
            if (tool.isTraining)
                return;

            m_LoadDefaultButton?.SetEnabled(false);
            m_ClearVenvButton?.SetEnabled(false);
            m_TrainButton?.SetEnabled(false);
            m_TerminateButton?.SetEnabled(true);

            ClearTrainingResults();

            await tool.Train();

            TryLoadTrainingResults();

            m_TerminateButton?.SetEnabled(false);
            m_TrainButton?.SetEnabled(true);
            m_LoadDefaultButton?.SetEnabled(true);

            UpdateClearVenvButtonState();
        }

        /// <summary>
        /// Terminate the training child process from Editor.
        /// </summary>
        private void OnClickTerminateButton()
        {
            if (!tool.isTraining)
                return;

            tool.Terminate();

            ClearTrainingResults();

            m_TerminateButton?.SetEnabled(false);
            m_TrainButton?.SetEnabled(true);
            m_LoadDefaultButton?.SetEnabled(true);

            UpdateClearVenvButtonState();
        }

        private IEnumerator ClearVenvCoroutine()
        {
            tool.ClearVenv();
            yield return null;

            UpdateClearVenvButtonState();
        }

        #endregion

        #region Training Progress Bar
        private void RenderProgressBar()
        {
            var progressInfo = tool.progressInfo;
            if (progressInfo.isTraining)
            {
                m_Progress.style.display = DisplayStyle.Flex;
                m_Progress.value = progressInfo.value;
                m_Progress.title = progressInfo.message;
            }
            else
            {
                m_Progress.style.display = DisplayStyle.None;
            }
        }
        #endregion

        #region Training Results UI

        /// <summary>
        /// Initialize Training Results UI.
        /// </summary>
        /// <param name="root"></param>
        private void InitTrainingResults(VisualElement root)
        {
            var visualTree = Resources.Load<VisualTreeAsset>("TrainerResult");
            visualTree.CloneTree(root);

            m_LossElement = root.Q("LossElement");

            m_LossFigure = m_LossElement.Q<Image>("LossFigure");
            m_LossFigure.scaleMode = ScaleMode.ScaleToFit;

            m_ModelNameField = root.Q<TextField>("ModelName");

            m_RevealLossButton = root.Q<Button>("RevealLossButton");
            m_RevealModelButton = root.Q<Button>("RevealModelButton");
            m_ClearResultButton = root.Q<Button>("ClearResultButton");

            // Reveal Loss Button
            m_RevealLossButton.RegisterCallback<ClickEvent>(evt =>
            {
                var fullLossPath = Path.GetFullPath(tool.lossPath);
                var fullAssetsPath = Path.GetFullPath(Application.dataPath);

                if (fullLossPath.StartsWith(fullAssetsPath))
                {
                    var revealPath = fullLossPath.Replace(fullAssetsPath, "Assets");
                    var lossTex = AssetDatabase.LoadAssetAtPath<Texture2D>(revealPath);
                    EditorGUIUtility.PingObject(lossTex);
                    Selection.activeObject = lossTex;
                    EditorUtility.FocusProjectWindow();
                }
                else
                {
                    EditorUtility.RevealInFinder(fullLossPath);
                }
            });

            // Reveal Model Button 
            m_RevealModelButton.RegisterCallback<ClickEvent>(evt =>
            {
                var fullModelPath = Path.GetFullPath(tool.modelPath);
                var fullAssetsPath = Path.GetFullPath(Application.dataPath);

                if (fullModelPath.StartsWith(fullAssetsPath))
                {
                    var revealPath = fullModelPath.Replace(fullAssetsPath, "Assets");
                    var modelAsset = AssetDatabase.LoadAssetAtPath<TextAsset>(revealPath);
                    if (modelAsset != null)
                    {
                        EditorGUIUtility.PingObject(modelAsset);
                        Selection.activeObject = modelAsset;
                        EditorUtility.FocusProjectWindow();
                    }
                }
                else
                {
                    EditorUtility.RevealInFinder(fullModelPath);
                }
            });

            // Clear Result Button
            m_ClearResultButton.RegisterCallback<ClickEvent>(evt =>
            {
                ClearTrainingResults();
            });

            TryLoadTrainingResults();
        }

        /// <summary>
        /// Try loading training results UI, including loss figure and onnx model. 
        /// </summary>
        /// <returns>(<see cref="bool"/>) The result flag indicating whether there are valid training results.</returns>
        private bool TryLoadTrainingResults()
        {
            var lossTex = LoadTexture2DAtPath(tool.lossPath);
            var success = lossTex != null && File.Exists(tool.modelPath);
            if (success)
            {
                m_LossElement.AddToClassList(k_LossLoadedStyle);
                m_LossFigure.image = lossTex;

                var modelFile = Path.GetFileName(tool.modelPath);
                m_ModelNameField.SetValueWithoutNotify(modelFile);
                m_ModelNameField.tooltip = modelFile;
                m_RevealLossButton.SetEnabled(true);
                m_RevealModelButton.SetEnabled(true);
                m_ClearResultButton.SetEnabled(true);
            }
            else
            {
                ClearTrainingResults();
            }
            return success;
        }

        /// <summary>
        /// Clear training results UI.
        /// </summary>
        private void ClearTrainingResults()
        {
            m_LossElement.RemoveFromClassList(k_LossLoadedStyle);
            m_LossFigure.image = null;

            m_ModelNameField.SetValueWithoutNotify(string.Empty);
            m_ModelNameField.tooltip = string.Empty;
            m_RevealLossButton.SetEnabled(false);
            m_RevealModelButton.SetEnabled(false);
            m_ClearResultButton.SetEnabled(false);

            tool.lossPath = null;
            tool.modelPath = null;
        }

        /// <summary>
        /// Load Texture2D from path.
        /// </summary>
        /// <param name="path">Full path to the picture file.</param>
        /// <returns>(<see cref="Texture2D"/>) The texture.</returns>
        private static Texture2D LoadTexture2DAtPath(string path)
        {
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
                return null;

            byte[] bytes = File.ReadAllBytes(path);
            if (bytes == null || bytes.Length == 0)
                return null;

            Texture2D tex = new Texture2D(2, 2);
            tex.hideFlags = HideFlags.HideAndDontSave;
            tex.LoadImage(bytes);
            return tex;
        }

        #endregion
    }
}
