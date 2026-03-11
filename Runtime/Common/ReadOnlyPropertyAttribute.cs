using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Tuanjie.NeuralDeformer
{
	[AttributeUsage(AttributeTargets.Field)]
	internal class ReadOnlyPropertyAttribute : PropertyAttribute { }

#if UNITY_EDITOR
	[CustomPropertyDrawer(typeof(ReadOnlyPropertyAttribute))]
    internal class ReadOnlyPropertyAttributeDrawer : PropertyDrawer
	{
		public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
		{
			return EditorGUI.GetPropertyHeight(property, label, true);
		}

		public override void OnGUI(Rect rect, SerializedProperty property, GUIContent label)
		{
			var enabled = GUI.enabled;
			GUI.enabled = false;
			EditorGUI.PropertyField(rect, property, label, true);
			GUI.enabled = enabled;
		}
	}
#endif
}
