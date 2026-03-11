using System.Collections;
using System.Collections.Generic;
using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEditor;
using System.Text.RegularExpressions;
using System.Linq;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text;

namespace Tuanjie.NeuralDeformer
{
    /// <summary>
    /// <see cref="NeuralDeformerTrainer"/> class manages training process of deformer neural network.
    /// </summary>
    [ExecuteInEditMode, RequireComponent(typeof(NeuralDeformerDatasetBuilder))]
    [AddComponentMenu("Neural Deformer/Neural Deformer Trainer")]
    public class NeuralDeformerTrainer : MonoBehaviour
    {
        /// <summary>
        /// Ways to initialize the center vectors in the RBF layer.
        /// </summary>
        public enum CenterInitType
        {
            /// <summary>
            /// Initialize by k-means clustering for all of samples from dataset.
            /// </summary>
            KMeans,

            /// <summary>
            /// Initialize by randomly sampling from uniform distribution of [-1, 1].
            /// </summary>
            Random
        }

        [Serializable]
        internal class ProgressInfo
        {
            public bool isTraining { get; private set; }
            public float value { get; private set; }
            public string message { get; private set; }
            public int id { get; private set; }

            public event Action onProgressUpdated;

            internal void StartProgress(string description = null)
            {
                isTraining = true;
                value = 0;
                message = description ?? string.Empty;
                id = -1;
                EditorApplication.update += UpdateBackgroundProgress;
                onProgressUpdated?.Invoke();
            }

            internal void ReportProgress(float progress, string description = null)
            {
                value = progress;
                message = description ?? string.Empty;
                onProgressUpdated?.Invoke();
            }

            internal void RemoveProgress()
            {
                isTraining = false;
                onProgressUpdated?.Invoke();
            }

            void UpdateBackgroundProgress()
            {
                if (isTraining)
                {
                    if (id < 0) id = Progress.Start(nameof(NeuralDeformerTrainer));
                    if (id >= 0 && Progress.Exists(id)) Progress.Report(id, value, message);
                }
                else
                {
                    if (id >= 0 && Progress.Exists(id)) id = Progress.Remove(id);
                    EditorApplication.update -= UpdateBackgroundProgress;
                }
            }
        }

        [Header("Hyper-paramters")]
        [Tooltip("Whether to use default run name, or it is specified by the user.")]
        public bool useDefaultRunName;

        [Tooltip("Custom run name.")]
        public string runName;

        [Tooltip("Random seed used in training process.")]
        public int seed;

        [Tooltip("Number of center vectors in the RBF layer."), Min(1f)]
        public int centersNumber;

        [Tooltip("Ways to initialize the center vectors. They are then optimized during training.")]
        public CenterInitType centersInit;

        [Tooltip("Betas' initial values which are then optimized during training.")]
        public float betasInit;

        [Tooltip("Training batch size."), Min(1f)]
        public int batchSize;

        [Tooltip("Shuffle training batch of not.")]
        public bool shuffle;

        [Tooltip("Total training epochs."), Range(1, 999)]
        public int epochs;

        [Tooltip("Initial learning rate."), Min(0f)]
        public float learningRate;

        [Tooltip("Task name, a.k.a. the name of deformation target object.")]
        public string taskName;

        [Tooltip("Dataset directory.")]
        public string datasetDir;

        [Header("Logging")]
        [Tooltip("Whether to save report to tensorboard.")]
        public bool enableReport;

        [Tooltip("Logs directory where tensorboard reports and loss figures are saved.")]
        public string reportDir;

        [Tooltip("Models directory where trained onnx models are saved.")]
        public string modelsDir;

        [Tooltip("Whether to print verbose info to console.")]
        public bool verbose;

        /// <summary>
        /// Default settings
        /// </summary>
        private const bool k_DefaultUseDefaultRunName = true;
        private const string k_DefaultRunName = "";
        private const int k_DefaultSeed = 0;
        private const int k_DefaultCentersNumber = 32;
        private const CenterInitType k_DefaultCentersInit = CenterInitType.KMeans;
        private const float k_DefaultBetasInit = 0.5f;
        private const int k_DefaultBatchSize = 8;
        private const bool k_DefaultShuffle = true;
        private const int k_DefaultEpochs = 300;
        private const float k_DefaultLearningRate = 1e-3f;
        private const bool k_DefaultEnableReport = false;
        private const string k_DefaultReportDirName = "logs";
        private const string k_DefaultModelsDirName = "models";
        private const string k_TuanjiePackageFile = "package.json";
        private const string k_PythonEntryFile = "Python~/train.py";
        private const string k_PythonVenvDir = "Python~/.venv";

        [SerializeField]
        internal ProgressInfo progressInfo = new();

        /// <summary>
        /// Full path to the loss figure (.png).
        /// </summary>
        private string _lossPath;
        /// <summary>
        /// Full path to the loss figure (.png).
        /// </summary>
        internal string lossPath { get => _lossPath; set => _lossPath = value; }

        private string _modelPath;
        /// <summary>
        /// Full path to the trained model (.onnx).
        /// </summary>
        internal string modelPath { get => _modelPath; set => _modelPath = value; }

        /// <summary>
        /// Reference to the <see cref="NeuralDeformerDatasetBuilder"/> component.
        /// </summary>
        private NeuralDeformerDatasetBuilder m_Data = null;

        internal NeuralDeformerDatasetBuilder data
        {
            get
            {
                if (m_Data == null)
                {
                    m_Data = gameObject.GetComponent<NeuralDeformerDatasetBuilder>();
                }
                return m_Data;
            }
        }

        internal string venvDir => Path.Combine(packageRootDirectory, k_PythonVenvDir).Replace("\\", "/");

        internal bool venvExists
        {
            get
            {
                return Directory.Exists(venvDir);
            }
        }

#if UNITY_EDITOR
        /// <summary>
        /// Important internal state variables.
        /// </summary>
        private bool m_Initialized = false;

        /// <summary>
        /// Current progress value, from 0 to 1.
        /// </summary>
        private float m_ProgressValue = 0;

        /// <summary>
        /// Process ID of the training process. 
        /// It is >= 0 when training is running, and -1 when not.
        /// </summary>
        private int m_Pid = -1;

        public bool isTraining => progressInfo.isTraining;

        private static string s_PackageRootDirectory = null;

        public static string packageRootDirectory
        {
            get
            {
                s_PackageRootDirectory ??= GetPackageRootDirectory();
                return s_PackageRootDirectory;
            }
        }

        private static string GetPackageRootDirectory([CallerFilePath] string currentPath = "")
        {
            string rootDir;

            for (rootDir = currentPath; !string.IsNullOrEmpty(rootDir); rootDir = Path.GetDirectoryName(rootDir))
            {
                string packageJsonPath = Path.Combine(rootDir, k_TuanjiePackageFile);

                if (File.Exists(packageJsonPath))
                {
                    break;
                }
            }

            Assert.IsTrue(!string.IsNullOrEmpty(rootDir), "Tuanjie package directory not found!");
            return rootDir;
        }

        /// <summary>
        /// Get the exact full path of the entry python script "train.py". Return null if not found.
        /// </summary>
        /// <returns></returns>
        private string GetEntryPath()
        {
            string entryPath = Path.Combine(packageRootDirectory, k_PythonEntryFile).Replace("\\", "/");

            Assert.IsTrue(File.Exists(entryPath), $"Entry python script not found at {entryPath}!");

            return Path.GetFullPath(entryPath);
        }

        public void ClearVenv()
        {
            if (venvExists)
            {
                Directory.Delete(venvDir, true);
            }

            UnityEngine.Debug.Log("Clear Python .venv done.");
        }

        void OnValidate()
        {
            if (m_Initialized)
                return;

            // Execute when this component is first attached to a GameObject in edit mode.
            m_Initialized = true;
            SyncDataset();
        }

        public void SyncDataset()
        {
            SyncFromDataProcessing();

            var rootDir = Path.GetDirectoryName(datasetDir);
            reportDir = Path.Combine(rootDir, k_DefaultReportDirName).Replace("\\", "/");
            modelsDir = Path.Combine(rootDir, k_DefaultModelsDirName).Replace("\\", "/");
        }

        /// <summary>
        /// Set default settings.
        /// </summary>
        public void ResetParameter()
        {
            useDefaultRunName = k_DefaultUseDefaultRunName;
            runName = k_DefaultRunName;
            seed = k_DefaultSeed;
            centersNumber = k_DefaultCentersNumber;
            centersInit = k_DefaultCentersInit;
            betasInit = k_DefaultBetasInit;
            batchSize = k_DefaultBatchSize;
            shuffle = k_DefaultShuffle;
            epochs = k_DefaultEpochs;
            learningRate = k_DefaultLearningRate;
            enableReport = k_DefaultEnableReport;
        }

        /// <summary>
        /// Synchronize settings from <see cref="NeuralDeformerDatasetBuilder"/> component.
        /// </summary>
        public void SyncFromDataProcessing()
        {
            taskName = data.taskName;
            datasetDir = data.datasetDir;
        }

        /// <summary>
        /// Ensure the directory exists. If it does not, create it.
        /// </summary>
        /// <param name="dirPath">The directory path.</param>
        /// <param name="isInAssets">The result flag indicates whether <paramref name="dirPath"/> is in current project's "Assets/" directory.</param>
        /// <returns>(<see cref="string"/>) The absolute/full path of <paramref name="dirPath"/>.</returns>
        private static string EnsureDirectoryExists(string dirPath, out bool isInAssets)
        {
            var fullDir = Path.GetFullPath(dirPath);
            var fullAssetDir = Path.GetFullPath(Application.dataPath);

            if (!Directory.Exists(fullDir))
            {
                Directory.CreateDirectory(fullDir);
            }

            isInAssets = fullDir.StartsWith(fullAssetDir);
            if (isInAssets)
            {
                AssetDatabase.Refresh();
            }

            return fullDir;
        }

        private bool GetNvidiaGPUInfo(out float driverVersion, out float computeCapability)
        {
            bool foundGPU = false;
            driverVersion = 0;
            computeCapability = 0;

            try
            {
                string nvidiaSmiPath = "nvidia-smi";
                if (!File.Exists("/usr/bin/nvidia-smi") && !File.Exists("C:\\Windows\\System32\\nvidia-smi.exe"))
                {
                    UnityEngine.Debug.LogWarning("[Train] NVIDIA-SMI not found. Skipping GPU detection.");
                    return false;
                }

                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = nvidiaSmiPath,
                    Arguments = "--query-gpu=name,driver_version,compute_cap --format=csv,noheader",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                string output, error;
                using (Process preProcess = new Process { StartInfo = psi })
                {
                    preProcess.Start();
                    output = preProcess.StandardOutput.ReadToEnd();
                    error = preProcess.StandardError.ReadToEnd();
                    preProcess.WaitForExit();
                }

                if (string.IsNullOrEmpty(error))
                {
                    using (StringReader reader = new StringReader(output))
                    {
                        string line;
                        while ((line = reader.ReadLine()) != null)
                        {
                            var parts = line.Split(',');
                            if (parts.Length >= 3)
                            {
                                if (float.TryParse(parts[1].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float parsedDriver) &&
                                    float.TryParse(parts[2].Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float parsedCC))
                                {
                                    foundGPU = true;
                                    driverVersion = parsedDriver;
                                    computeCapability = parsedCC;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogWarning($"[Train] Failed to get NVIDIA GPU info: {ex.Message}");
                foundGPU = false;
            }

            return foundGPU;
        }

        private static readonly Dictionary<string, int> s_DriverRequirement = new Dictionary<string, int>()
        {
            { "cu128", 560 },
            { "cu126", 555 },
            { "cu124", 550 },
            { "cu118", 520 },
        };

        private enum TorchBackendType
        {
            CPU,
            CPU_MAC_APPLE_SILICON,
            CPU_MAC_INTEL,
            CUDA
        }

        private string ResolveCudaVersion(float cc, float driver)
        {
            if (cc >= 9.0f && driver >= s_DriverRequirement["cu128"]) return "cu128";
            if (cc >= 8.9f && driver >= s_DriverRequirement["cu126"]) return "cu126";
            if (cc >= 8.0f && driver >= s_DriverRequirement["cu124"]) return "cu124";
            if (cc >= 7.0f && driver >= s_DriverRequirement["cu118"]) return "cu118";
            return "cpu";
        }

        private (TorchBackendType backend, string version) DetectTorchBackend()
        {
            string os = SystemInfo.operatingSystem.ToLowerInvariant();
            string arch = SystemInfo.processorType.ToLowerInvariant();

            if (os.Contains("mac"))
            {
                bool isAppleSilicon = arch.Contains("arm") || arch.Contains("apple");
                if (verbose) UnityEngine.Debug.Log($"[Train] macOS { (isAppleSilicon ? "Apple Silicon" : "Intel") } detected.");
                return (isAppleSilicon ? TorchBackendType.CPU_MAC_APPLE_SILICON : TorchBackendType.CPU_MAC_INTEL, "cpu");
            }

            if (!GetNvidiaGPUInfo(out float driver, out float cc))
            {
                UnityEngine.Debug.LogWarning("[Train] No compatible NVIDIA GPU found. Using CPU backend for PyTorch.");
                return (TorchBackendType.CPU, "cpu");
            }

            string cudaVersion = ResolveCudaVersion(cc, driver);
            if (cudaVersion == "cpu")
            {
                UnityEngine.Debug.LogWarning($"[Train] GPU driver ({driver}) or compute capability ({cc}) unsupported. Using CPU backend for PyTorch.");
                return (TorchBackendType.CPU, "cpu");
            }

            if (verbose) UnityEngine.Debug.Log($"[Train] CUDA backend selected for PyTorch: {cudaVersion} (Driver: {driver}, CC: {cc})");
            return (TorchBackendType.CUDA, cudaVersion);
        }

        /// <summary>
        /// Get directories to search for the uv executable depending on current OS.
        /// </summary>
        private IEnumerable<string> GetUvSearchDirectories()
        {
            var directories = new List<string>();

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            bool isWindows = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;
            bool isOSX = Application.platform == RuntimePlatform.OSXEditor || Application.platform == RuntimePlatform.OSXPlayer;

            if (isOSX)
            {
                directories.Add("/opt/homebrew/bin");
                directories.Add("/usr/local/bin");
                directories.Add("/usr/bin");
                if (!string.IsNullOrEmpty(home))
                {
                    directories.Add(Path.Combine(home, ".local", "bin"));
                    directories.Add(Path.Combine(home, ".cargo", "bin"));
                }
            }
            else if (isWindows)
            {
                if (!string.IsNullOrEmpty(localAppData))
                {
                    directories.Add(Path.Combine(localAppData, "uv"));
                    directories.Add(Path.Combine(localAppData, "Programs", "uv"));
                }
                if (!string.IsNullOrEmpty(home))
                {
                    directories.Add(Path.Combine(home, ".local", "bin"));
                    directories.Add(Path.Combine(home, ".cargo", "bin"));
                    directories.Add(Path.Combine(home, "scoop", "shims"));
                }
                var pf = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
                var pfx86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
                if (!string.IsNullOrEmpty(pf)) directories.Add(pf);
                if (!string.IsNullOrEmpty(pfx86)) directories.Add(pfx86);
                directories.Add(@"C:\\Windows\\System32");
            }
            else
            {
                // Linux
                directories.Add("/usr/local/bin");
                directories.Add("/usr/bin");
                directories.Add("/bin");
                if (!string.IsNullOrEmpty(home))
                {
                    directories.Add(Path.Combine(home, ".local", "bin"));
                    directories.Add(Path.Combine(home, ".cargo", "bin"));
                }
            }

            return directories;
        }

        /// <summary>
        /// Try to resolve an absolute path to the "uv" executable across common install locations.
        /// Falls back to just "uv" if not found, allowing system PATH to resolve it.
        /// </summary>
        /// <returns>Absolute path to uv if found; otherwise, "uv".</returns>
        private string ResolveUvExecutablePath()
        {
            try
            {
                var dirs = GetUvSearchDirectories();
                bool isWindows = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;
                string uvName = isWindows ? "uv.exe" : "uv";

                foreach (var dir in dirs)
                {
                    try
                    {
                        if (string.IsNullOrWhiteSpace(dir))
                            continue;

                        var direct = Path.Combine(dir, uvName);
                        var nested = Path.Combine(dir, "uv", uvName);
                        if (File.Exists(direct)) return direct;
                        if (File.Exists(nested)) return nested;
                    }
                    catch { }
                }
            }
            catch { }

            return "uv"; // Let PATH resolve it
        }

        private void SetPythonRequirements()
        {
            var (backend, torchCudaVersion) = DetectTorchBackend();
            string backendName = backend.ToString().ToLowerInvariant();

            string torchIndex = backend switch
            {
                TorchBackendType.CUDA => $"https://download.pytorch.org/whl/{torchCudaVersion}",
                TorchBackendType.CPU  => "https://download.pytorch.org/whl/cpu",
                TorchBackendType.CPU_MAC_INTEL  => "https://download.pytorch.org/whl/cpu",
                _ => ""
            };

            string torchVersion = backend switch
            {
                TorchBackendType.CPU_MAC_INTEL  => "torch==2.2.0",
                // _ => "torch>=2.6.0"
                _ => "torch==2.2.0",
            };

            StringBuilder pyprojectContent = new StringBuilder($@"[build-system]
requires = [""setuptools>=68"", ""wheel""]
build-backend = ""setuptools.build_meta""

[project]
name = ""neural_deformer_trainer""
version = ""1.0.0""
description = ""Neural Deformer Trainer: A neural network model that can predict accurate outfit deformations on 3D characters in motion.""
readme = ""README.md""
requires-python = "">=3.10,<3.13""
dependencies = [
    ""numpy<2"",
    ""ipykernel>=6.29.5"",
    ""matplotlib>=3.10.1"",
    ""onnx>=1.17.0"",
    ""onnxscript>=0.2.5"",
    ""tensorboardx>=2.6.2.2"",
    ""{torchVersion}"",
    ""torch-kmeans>=0.2.0""
]
");

            // add torchIndex only for CUDA & CPU
            if (!string.IsNullOrEmpty(torchIndex))
            {
                pyprojectContent.Append($@"
[[tool.uv.index]]
name = ""pytorch-{backendName}""
url = ""{torchIndex}""
explicit = true
");
            }

            // add default mirror source
            pyprojectContent.Append($@"
[[tool.uv.index]]
url = ""https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple""
");

            string directory = Path.GetDirectoryName(GetEntryPath());
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            string pyprojectPath = Path.Combine(directory, "pyproject.toml");
            var utf8NoBom = new UTF8Encoding(false);
            File.WriteAllText(pyprojectPath, pyprojectContent.ToString(), utf8NoBom);

            if (verbose) UnityEngine.Debug.Log($"[Train] pyproject.toml generated for backend: {backendName}");
        }

        /// <summary>
        /// Load arguments for the training process.
        /// </summary>
        /// <param name="fullDatasetDir">The full path of dataset directory.</param>
        /// <param name="fullModelsDir">The full path of models directory.</param>
        /// <param name="fullReportDir">The full path of report directory.</param>
        /// <returns></returns>
        private string LoadArguments(string fullDatasetDir, string fullModelsDir, string fullReportDir)
        {
            var args = new StringBuilder();
            if (!useDefaultRunName)
            {
                args.Append($"--run-name {runName} ");
            }
            args.Append($"--seed {seed} ");
            args.Append($"--centers {centersNumber} ");
            args.Append($"--centers-init {centersInit.ToString().ToLower()} ");
            args.Append($"--betas-init {betasInit} ");
            args.Append($"--batch-size {batchSize} ");
            if (!shuffle)
            {
                args.Append("--no-shuffle ");
            }
            args.Append($"--epochs {epochs} ");
            args.Append($"--lr {learningRate} ");
            args.Append($"--import-dir \"{fullDatasetDir}\" ");
            args.Append($"--export-dir \"{fullModelsDir}\" ");
            args.Append($"--report-dir \"{fullReportDir}\" ");
            if (!enableReport)
            {
                args.Append("--no-report ");
            }

            return args.ToString();
        }

        public async Task Train()
        {
            try
            {
                progressInfo.StartProgress("Start training...");
                await DoTrain();
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogException(e);
            }
            finally
            {
                progressInfo.RemoveProgress();
            }
        }

        /// <summary>
        /// Main entrance to start training a deformer neural network.
        /// </summary>
        /// <returns></returns>
        public async Task DoTrain()
        {
            SetPythonRequirements();

            // Manage entry python script and working directory.
            var pythonPath = GetEntryPath();
            var workingDir = Path.GetDirectoryName(pythonPath);

            // Manage input and output directories.
            var fullDatasetDir = EnsureDirectoryExists(Path.Combine(datasetDir, taskName), out _);
            var fullModelsDir = EnsureDirectoryExists(Path.Combine(modelsDir, taskName), out var isModelsInAssets);
            var fullReportDir = EnsureDirectoryExists(Path.Combine(reportDir, taskName), out var isReportInAssets);

            // Compose final command.
            var program = ResolveUvExecutablePath();
            var arguments = $"run --quiet \"{pythonPath}\" {LoadArguments(fullDatasetDir, fullModelsDir, fullReportDir)}";
            if (verbose) UnityEngine.Debug.Log($"[Train] Run command in Background Tasks: {program} {arguments}");

            // An indicator variable that asynchronously monitors the end of training child process.
            TaskCompletionSource<bool> eventHandled = new();

            // Start background progress indicator.
            m_ProgressValue = 0;
            progressInfo.ReportProgress(m_ProgressValue, "Configuring Python environment...");
            
            // Spawn a new child process.
            using Process process = new();
            {
                try
                {
                    // Set up the child process start info.
                    process.StartInfo = new ProcessStartInfo()
                    {
                        FileName = program,
                        Arguments = arguments,
                        WorkingDirectory = workingDir,
                        CreateNoWindow = true,              // Run in background.
                        UseShellExecute = false,
                        RedirectStandardError = true,       // Enable error redirection.
                        RedirectStandardOutput = true,      // Enable output redirection.
                        EnvironmentVariables =
                        {
                            // No buffering for outputs and errors to enable instant output display.
                            { "PYTHONUNBUFFERED", "1" },
                        
                            // Ignore future warnings, which will come out during onnx export.
                            { "PYTHONWARNINGS", "ignore::FutureWarning" },
                        }
                    };

                    // Register event handlers for output and errors.
                    process.OutputDataReceived += new DataReceivedEventHandler(OnReceiveOutputData);
                    process.ErrorDataReceived += new DataReceivedEventHandler(OnReceiveErrorData);

                    // Enable and register the Exited event, which will be triggered when the process exits.
                    process.EnableRaisingEvents = true;
                    process.Exited += new EventHandler((sender, e) =>
                    {
                        int exitCode = process.ExitCode;
                        if (verbose) UnityEngine.Debug.Log($"[Train] Process exited with code: {exitCode}");

                        eventHandled.TrySetResult(exitCode == 0);
                        m_Pid = -1;
                    });

                    // Start the child process to train the model via python.
                    Assert.IsTrue(process.Start(), "Fail to start the process.");
                    m_Pid = process.Id;
                    if (verbose) UnityEngine.Debug.Log($"[Train] Process started with pid: {m_Pid}");

                    // Start reading output and error streams asynchronously.
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                }
                catch (Exception)
                {                   
                    eventHandled.TrySetResult(false);
                    m_Pid = -1;

                    bool isWindows = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer;
                    bool isOSX = Application.platform == RuntimePlatform.OSXEditor || Application.platform == RuntimePlatform.OSXPlayer;
                    bool isLinux = Application.platform == RuntimePlatform.LinuxEditor || Application.platform == RuntimePlatform.LinuxPlayer;

                    string tips;
                    if (isOSX)
                    {
                        tips = "Tips (macOS):\n- Install uv from: https://docs.astral.sh/uv/getting-started/installation/\n- If installed via Homebrew: /opt/homebrew/bin/uv, ensure /opt/homebrew/bin and /usr/local/bin are in PATH.";
                    }
                    else if (isWindows)
                    {
                        tips = "Tips (Windows):\n- Install uv from: https://docs.astral.sh/uv/getting-started/installation/\n- Typical installs: %LOCALAPPDATA%\\uv\\uv.exe or %USERPROFILE%\\.local\\bin\\uv.exe, ensure these directories are in PATH.";
                    }
                    else if (isLinux)
                    {
                        tips = "Tips (Linux):\n- Install uv from: https://docs.astral.sh/uv/getting-started/installation/\n- Typical installs: ~/.local/bin/uv or /usr/local/bin/uv\n- Ensure these are in PATH (use ':' as separator).";
                    }
                    else
                    {
                        tips = "Tips:\n- Install uv from: https://docs.astral.sh/uv/getting-started/installation/\n- Ensure uv is in your system PATH.";
                    }

                    EditorUtility.DisplayDialog(
                        "NeuralDeformerTrainer Error",
                        "The command 'uv' is not executable. Ensure uv is installed and available in PATH.\n\n" + tips,
                        "OK");
                }

                // Asynchronously wait until the indicator gets a result.
                await Task.WhenAll(eventHandled.Task);
            }

            // Refesh AssetDatabase if the model or report are saved in Assets/.
            if (isModelsInAssets || isReportInAssets)
            {
                AssetDatabase.Refresh();
            }
        }

        // [Prefix] ...
        private static readonly Regex prefixRegex = new Regex(@"\[(.*?)\]\s*(.+)", RegexOptions.Compiled);

        // [Epoch xxx/yyy] ...
        private static readonly Regex epochRegex = new Regex(@"Epoch\s+(\d+)/(\d+)", RegexOptions.Compiled);

        /// <summary>
        /// Receive output data from the training child process.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void OnReceiveOutputData(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                var prefixMatch = prefixRegex.Match(e.Data);
                if (prefixMatch.Success)
                {
                    string prefix = prefixMatch.Groups[1].Value.Trim();

                    var epochMatch = epochRegex.Match(prefix);
                    if (epochMatch.Success)
                    {
                        // Update current progress value.
                        prefix = "Epoch";
                        if (int.TryParse(epochMatch.Groups[1].Value, out int currEpoch) &&
                            int.TryParse(epochMatch.Groups[2].Value, out int totalEpoch))
                        {
                            m_ProgressValue = (float)currEpoch / totalEpoch;
                        }
                    }

                    if (prefix == "Summary")
                    {
                        // Fetch the full path of loss figure and onnx model.
                        if (e.Data.Contains("loss figure saved to path: "))
                        {
                            lossPath = e.Data.Trim().Split("loss figure saved to path: ").Last();
                        }
                        else if (e.Data.Contains("model saved to path: "))
                        {
                            modelPath = e.Data.Trim().Split("model saved to path: ").Last();
                        }
                    }
                }

                // Ensure API calls are on main thread
                EditorApplication.delayCall += () =>
                {
                    if (verbose) UnityEngine.Debug.Log($"[Train] {e.Data}");
                    progressInfo.ReportProgress(m_ProgressValue, e.Data);
                };
            }
        }

        /// <summary>
        /// Receive error data from the training child process.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void OnReceiveErrorData(object sender, DataReceivedEventArgs e)
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                var prefixMatch = prefixRegex.Match(e.Data);
                if (prefixMatch.Success)
                {
                    string prefix = prefixMatch.Groups[1].Value.Trim();
                    if (prefix == "Exception")
                    {
                        string jsonText = prefixMatch.Groups[2].Value.Trim();
                        try
                        {
                            var error = JsonUtility.FromJson<PythonRuntimeExceptionInfo>(jsonText);
                            if (error.Valid)
                            {
                                var ex = new PythonRuntimeException(error.type, error.message, error.traceback);
                                EditorApplication.delayCall += () =>
                                {
                                    UnityEngine.Debug.LogException(ex);
                                };
                                return;
                            }
                        }
                        catch (Exception ex)
                        {
                            EditorApplication.delayCall += () =>
                            {
                                UnityEngine.Debug.LogException(ex);
                            };
                        } 
                    }
                }

                // Ensure API calls are on main thread
                EditorApplication.delayCall += () =>
                {
                    UnityEngine.Debug.LogError($"[Train] {e.Data}");
                };
            }
        }

        /// <summary>
        /// Terminate the training child process.
        /// </summary>
        public void Terminate()
        {
            if (isTraining)
            {
                try
                {
                    var process = Process.GetProcessById(m_Pid);
                    if (process != null && !process.HasExited)
                    {
                        process.Kill();
                        process.WaitForExit(1000); // Wait up to 1 second for process to exit
                        process.Dispose();
                    }
                }
                catch (ArgumentException)
                {
                    // Process already exited or doesn't exist
                    UnityEngine.Debug.LogWarning("[Train] Process already terminated or doesn't exist.");
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.LogError($"[Train] Error terminating process: {ex.Message}");
                }
                finally
                {
                    m_Pid = -1;
                    progressInfo.RemoveProgress();
                }
            }
        }

        private struct PythonRuntimeExceptionInfo
        {
            public string type;
            public string message;
            public string traceback;
            public bool Valid => !string.IsNullOrEmpty(type) && !string.IsNullOrEmpty(message) && !string.IsNullOrEmpty(traceback);
        }

        private class PythonRuntimeException : Exception
        {
            public PythonRuntimeException(string type, string message, string traceback)
                : base($"[{type}] {message}\n{traceback}") { }
        }
#endif
    }
}