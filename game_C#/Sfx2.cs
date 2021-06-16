using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Sfx2 : MonoBehaviour
{
    static AudioSource audioSource;
    public static AudioClip audioClip;
    void Start()
    {
        audioSource = GetComponent<AudioSource>();
        audioClip = Resources.Load<AudioClip>("Fall");
    }

    // Update is called once per frame
    public static void SoundPlay()
    {
        audioSource.PlayOneShot(audioClip);
    }
}

