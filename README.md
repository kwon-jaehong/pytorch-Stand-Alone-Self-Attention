# pytorch-Stand-Alone-Self-Attention
--------------------------------------------
논문 주소 https://arxiv.org/abs/1906.05909

--------------------

Stand-alone self-attention in vision
->컨볼루션 레이어대신 local self attention layer로 대체하겠다
-> 파라미터수 감소

컨볼루션 연산은 큰 receptive field에 대한 스케일링 특성이 좋지 않기 때문에....
다시 말해 컨볼루젼연산이 지역적(localiztion)이기 때문에 long range interaction(아주 긴 거리의 픽셀간 관계)을 캡쳐하기에는 어렵다는 한계점을 가진다.


이논문은 local 어텐션이다. (https://www.youtube.com/watch?v=U-ME2h-ezJM&t=3960s) 참조

self-attension이란 ? 입력데이터 -> 쿼리,키,벨류 / 데이터 내의 상관 관계를 바탕으로 특징을 추출함
셀프어텐션은 CNN 필터와 다르게 픽셀간의 상관관계를 구할 수 있음
-> https://www.youtube.com/watch?v=bgsYOGhpxDc&t=497s


--------------
key와 values는 비슷한 비율?의 값이다
-> 어짜피 스칼라만 다르지 원본(입력값)에 대해 뭔가 곱해서 나오는 값이기 때문


쿼리와 키는 각각 요소에 단순 곱셈
쿼리가 0.3이고 키가 3x3 의 원행렬이면 싹다 0.3으로 되고, 거기에 소프트 맥스


벨류와 소프트 맥스는 단순 내적 (곱의 합)


컨볼루션 스템(stem)이란? 
레즈넷 같은 경우 앞단에 7x7 컨볼루션을 실행하여 shallow한 특징을 뽑을수 있다?
-> 초장에 크게 특징을 추출하겟다
-> 에지와 같은 중요정보를 추출하는 기능을 함(지역적 특징), 기존 컨볼루션 레이어는 뒤로갈수록 글로벌한 특징을 추출 함
앞에 큰 사이즈의 커널사이즈를 가지는 컨볼루션 연산을 스템(stem)이라 한다


Stand-alone self-attention논문에서는 어텐션 스템(stem)도 제안함


-------------

단순 커널연산의 파라미터
로컬 셀프 어텐션 파라미터 갯수












------------















논문 한글 해석 블로그
1. https://2bdbest-ds.tistory.com/entry/Attention-Stand-Alone-Self-Attention-in-Vision-ModelsNIPS-2019
2. https://ffighting.tistory.com/entry/Stand-alone-self-attention-in-vision-models-%ED%95%B5%EC%8B%AC-%EB%A6%AC%EB%B7%B0
3. https://www.youtube.com/watch?v=6hadVw4Sy2M&t=744s

어텐션 관련
4. https://ys-cs17.tistory.com/46
5. https://www.youtube.com/watch?v=U-ME2h-ezJM&t=3960s


------------------


레즈넷 26기반의 어텐션

쥬피터 노트북 기반

데이터셋 사이퍼 10 

참고 사이트
