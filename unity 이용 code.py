using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class HandAnimation : MonoBehaviour
{
    public Transform thumbTip;
    public Transform indexTip;
    public Transform middleTip;
    public Transform ringTip;
    public Transform pinkyTip;

    public TextAsset handDataCSV;
    private List<float> angles_12_23 = new List<float>();
    private List<float> angles_23_34 = new List<float>();
    private List<Vector3> landmark_12 = new List<Vector3>();
    private List<Vector3> landmark_23 = new List<Vector3>();
    private List<Vector3> landmark_34 = new List<Vector3>();

    private int currentFrame = 0;

    void Start()
    {
        LoadHandData();
        StartCoroutine(PlayHandAnimation());
    }

    void LoadHandData()
    {
        string[] data = handDataCSV.text.Split('\n');
        for (int i = 1; i < data.Length; i++) // Skip the header row
        {
            string[] row = data[i].Split(',');
            angles_12_23.Add(float.Parse(row[1]));
            angles_23_34.Add(float.Parse(row[2]));
            landmark_12.Add(new Vector3(float.Parse(row[3]), float.Parse(row[4]), float.Parse(row[5])));
            landmark_23.Add(new Vector3(float.Parse(row[6]), float.Parse(row[7]), float.Parse(row[8])));
            landmark_34.Add(new Vector3(float.Parse(row[9]), float.Parse(row[10]), float.Parse(row[11])));
        }
    }

    IEnumerator PlayHandAnimation()
    {
        while (currentFrame < angles_12_23.Count)
        {
            thumbTip.position = landmark_12[currentFrame];
            indexTip.position = landmark_23[currentFrame];
            middleTip.position = landmark_34[currentFrame];

            yield return null; // Wait for the next frame
            currentFrame++;
        }
    }
}
