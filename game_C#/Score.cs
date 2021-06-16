using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class Score : MonoBehaviour
{
    public int Respawn1;
    Text text;
    public static int coinAmount;
    // Start is called before the first frame update
    void Start()
    {
        text = GetComponent<Text>();
    }

    // Update is called once per frame
    void Update()
    {
        text.text = coinAmount.ToString();

        if(coinAmount==7)
        {
            Score.coinAmount = 0;
            SceneManager.LoadScene(Respawn1);
        }
    }

    


}
