using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine.SceneManagement;

public class Player : MonoBehaviour
{
    Rigidbody2D rb;
    //public int Respawn;
    float moveX;
    float a = 2;

    Thread receiveThread;
    UdpClient client;
    int port;

    bool goddess;
    bool tree;
    bool warrior2;

    [SerializeField] [Range(100f, 800f)] float moveSpeed = 600f;
    [SerializeField] [Range(100f, 800f)] float jumpForce = 600f;

    int playerLayer, groundLayer;

    // Start is called before the first frame update
    void Start()
    {
        port = 5065;
        InitUDP();

        rb = GetComponent<Rigidbody2D>();

        playerLayer = LayerMask.NameToLayer("Player");
        groundLayer = LayerMask.NameToLayer("Ground");
    }

    private void InitUDP()
    {
        print("UDP Initialized");

        receiveThread = new Thread(new ThreadStart(ReceiveData)); //1 
        receiveThread.IsBackground = true; //2
        receiveThread.Start(); //3
    }

    private void ReceiveData()
    {
        client = new UdpClient(port); //1
        while (true) //2
        {
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Parse("0.0.0.0"), port); //3
            byte[] data = client.Receive(ref anyIP); //4

            string text = Encoding.UTF8.GetString(data); //5
            //print(">> " + text);

            if (text == "goddess")
            {
                //Debug.Log(text);
                print(">> " + text);
                goddess = true;
            }
            else if (text == "warrior2")
            {
                //Debug.Log(text);
                print(">> " + text);
                tree = true;
            }
            else if (text == "tree")
            {
                //Debug.Log(text);
                print(">> " + text);
                warrior2 = true;
            }
            //jump = true; //6
        }
    }

    // Update is called once per frame
    void Update()
    {
        //moveX = Input.GetAxis("Horizontal") * moveSpeed * Time.deltaTime;
        //rb.velocity = new Vector2(moveX, rb.velocity.y);
        /*
        if (Input.GetButtonDown("Jump"))
        {
            if (rb.velocity.y == 0)
                rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Force);
        }
        */

        if (goddess == true)
        {
            gameObject.transform.position += new Vector3(-(a + 2), a + 1);
            goddess = false;
        }
        else if (tree == true)
        {
            gameObject.transform.position += new Vector3(a + 2, a + 1);
            tree = false;
        }
        else if (warrior2 == true)
        {
            gameObject.transform.position += new Vector3(0, a + 1);
            warrior2 = false;
        }

        if (rb.velocity.y > 0)
            Physics2D.IgnoreLayerCollision(playerLayer, groundLayer, true);
        else
            Physics2D.IgnoreLayerCollision(playerLayer, groundLayer, false);


    }
    
}
