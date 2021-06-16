# AIot
# AIot 치매 예방 놀이 치료 게임

AIot 치매 예방 놀이 치료 게임으로 라즈베리파이 카메라에서 동작을 인식하여 움직이는 게임입니다.

[DeepLearning 폴더]
- [skeleton.py] : openpose를 사용하여 skeleton을 뽑는 코드
- [4c_학습.py] : 데이터 셋을 이용하여 3가지 포즈와 false 값으로 분류하는 학습 코드
- [model.py] : 학습한 모델 저장 후 모델의 성능을 평가한 코드
- [test_img.py] : 저장된 모델을 사용하여 테스트 이미지 하나로 추론 값을 확인하는 코드

[Rasberry.py]
라즈베리파이에서 opencv를 사용하여 카메라 값을 받고, TCP 통신으로 PC에게 전달하는 코드
(TCP Client)

[inference.py]
라즈베리파이 카메라 값을 받아 추론을 한 후 유니티(C#)로 추론 값 전달하는 코드
(TCP/UDP Server)

[game_C# 폴더]
- [Coin.cs] : 코인을 먹었을 때 상태 코드
- [KillPlayer.cs] : player가 떨어졌을 때 상태 코드
- [Player.cs] : 추론 값을 받아 player를 움직이는 코드 (UDP Client)
- [Score.cs] : score를 쌓는 코드
- [Sfx.cs] : 사운드 코드 1
- [Sfx2.cs] : 사운드 코드 2
