using System.IO;
using UnityEngine;
using UnityEditor;

namespace Tuanjie.NeuralDeformer
{
    [InitializeOnLoad]
    static class NeuralDeformerIconManager
    {
        static readonly string[] s_NeuralDeformerTypes = new string[]
        {
            "NeuralDeformerDatasetBuilder",
            "NeuralDeformerTrainer",
            "NeuralDeformerPlayer",
        };

        static readonly string s_NeuralDeformerIconFile = "NeuralDeformerIcon";

        static NeuralDeformerIconManager()
        {
            EditorApplication.delayCall += TrySetIconsForNeuralDeformerComponents;
            AssetDatabase.importPackageCompleted += _ => TrySetIconsForNeuralDeformerComponents();
        }

        private static void TrySetIconsForNeuralDeformerComponents()
        {
            foreach (var type in s_NeuralDeformerTypes)
            {
                SetIcon(type);
            }
        }

        private static void SetIcon(string monoType)
        {
            var icon = Resources.Load<Texture2D>(Path.Combine("Icons", s_NeuralDeformerIconFile));
            if (icon == null) return;

            var guids = AssetDatabase.FindAssets($"{monoType} t:MonoScript");
            foreach (var guid in guids)
            {
                var path = AssetDatabase.GUIDToAssetPath(guid);
                var script = AssetDatabase.LoadAssetAtPath<MonoScript>(path);
                if (script != null)
                {
                    EditorGUIUtility.SetIconForObject(script, icon);
                }
            }
        }
    }
}