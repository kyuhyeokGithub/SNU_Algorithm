import glob
import random
import time

def read(x):
    """
    x 를 open
    n : Array A의 길이
    A : Array A
    i : 찾고자하는 i-th smallest element에서 i
    만약 찾지못하면 0,0,0 리턴
    """
    try :
        f = open(x,'r')
        lines = f.readlines()
        f.close()
        n = int(lines[0].strip('\n'))
        A = list(map(int, lines[1].strip('\n').split()))
        i = int(lines[2].strip('\n'))
    except :
        n,A,i=0,0,0
    return n, A, i

def partition(n,A,x):
    """
    :param n: Length of list A
    :param A: list A
    :param x: partition의 기준이 될 원소의 index
    :return: A[x]를 기준으로 작은 원소들의 모임, 같은 원소들의 모임, 큰 원소들의 모임을 small, same, big으로 정의하고 이들을 리턴
    """
    # x의 위치에 있는 A[x]를 pivot이라 하자
    A[0],A[x]=A[x],A[0] # pivot을 list의 맨앞으로 이동, 분류할 원소들은 A[1],...,A[n-1]
    i,j,k=1,1,n-1
    # i : pivot과 비교하고자 하는 element 위치
    # j : j-1은 pivot보다 작은 elements들을 쌓았을때의 위치이고, j는 그 다음 pivot대비 small element가 앞으로 쌓일 위치
    # k : k+1은 pivot보다 큰 elements들을 뒤에서부터 쌓았을때의 위치이고, k는 그 다음 pivot대비 big element가 앞으로 쌓일 위치
    while(i<=k):
        if A[0]>A[i]:
            A[i],A[j]=A[j],A[i]
            i+=1
            j+=1
        elif A[0]<A[i]:
            A[i],A[k]=A[k],A[i]
            k-=1
        else :
            i+=1
    A[j-1],A[0]=A[0],A[j-1]
    # pivot보다 큰 원소들 list 뽑기
    if k==n-1: big=[] # pivot보다 큰 원소들이 없으므로 [] 리턴
    else : big=A[k+1:]
    # pivot보다 작은 원소들 list 뽑기
    if j==1:small=[] # pivot보다 작은 원소들이 없으므로 [] 리턴
    else : small=A[:j-1]
    # pivot랑 같은 원소들 list 뽑기
    if j-1>k:same=[] # pivot과 같은 원소들이 없으므로 [] 리턴
    else : same=A[j-1:k+1]
    return small, same, big

def insertion_sort(n,A):
    """
    각 원소를 돌며 앞의 원소와 자기를 비교하고
    자기가 작다면 자리 교환 후
    다시 그 앞의 원소와 자신을 비교
    해당 작업을 진행하면 작은순서대로 정렬
    """
    if n==1:
        return A
    for j in range(1,n,1):
        for i in range(j,0,-1):
            if A[i-1]>A[i] :
                A[i],A[i-1]=A[i-1],A[i]
    return A


def randomized_selection(n,A,i):
    """
    :param n: Length of list A
    :param A: list A
    :param i: i-th smallest element
    """
    if n==1 :
        return A[0]
    else :
        pivot_index = random.randint(0, n-1) # 0과 n-1 사이에서 임의로 pivot_index 설정
        small,same,big=partition(n,A,pivot_index) # pivot index를 기준으로 small, same, big element로 나눔
        if i<=len(small) : # 찾고자 하는 i가 small내에 존재
            return randomized_selection(len(small),small,i)
        elif i>len(small)+len(same) : # 찾고자 하는 i가 big내에 존재
            return randomized_selection(len(big),big,i-len(small)-len(same))
        else : # 그외의 경우는 pivot_index 위치의 원소와 값이 같으므로 same을 타겟팅
            return same[0]

def divide_size5(n,A):
    """
    length n의 Array A를 앞에서부터 5개씩 하나의 리스트로 묶고 이를 one element로 저장
    그들을 모아 리턴
    Ex.
    A=[(1,0),(2,1),(3,2),(4,3),(5,4),(6,5)]
    divide_group = [[(1,0),(2,1),(3,2),(4,3),(5,4)],[(6,5)]]
    """
    divide_group = []
    for i in range(0,n,5):
        divide_group.append(A[i:i+5])
    return divide_group

def medians(divide_target):
    """
    :param divide_target: 각 element는 하나의 리스트로 구성되어있으며 리스트는 숫자를 저장하고있고 각 리스트의 최대크기는 5이다
    :return: 각 element가 가지고있는 리스트에서 median을 뽑아 저장하며, 그 median들을 모아서 리스트로 리턴
    """
    median=[]
    for group in divide_target:
        l=len(group)
        group = insertion_sort(l,group)
        median.append(group[int((l-1)/2)])
    return median

def find_median(n,A):
    """
    :param n: list A의 길이
    :param A: list A의 원소는 (element, index)로 구성되어있음
    """
    if n<=5 :
        # len가 5이하의 짧은 경우, insertion sort를 하고 median return
        target = insertion_sort(n,A)
        return target[int((n-1)/2)]
    else :
        divide_A=divide_size5(n,A) # A를 앞에서부터 5개씩 하나의 그룹으로 묶음
        median_group=medians(divide_A) # 5개씩 묶은 그룹별로 median 원소들을 뽑아서 리턴
        return find_median(len(median_group),median_group) # median 원소들을 묶은 것을 가지고 다시 find_median 진행

def with_index(n,A):
    """
    리턴값인 list A_with_index의 i-th element는 다음과 같이 구성
    (A[i], i)
    Ex : A = [10,30,50], A_with_index = [(10,0),(30,1),(50,2)]
    """
    A_with_index = []
    for i in range(n):
        A_with_index.append((A[i],i))
    return A_with_index

def deterministic_selection(n,A,i):
    """
    :param n: Length of list A
    :param A: list A
    :param i: i-th smallest element
    """
    if n<=5 : # list 길이가 5이하면 insertion sort 실행하고 i-th element 리턴
        A=insertion_sort(n,A)
        return A[i-1]
    else :
        A_with_index = with_index(n,A) # A의 원소가 (A[i],i) 형식으로 구성되게 변경
        pivot_index = find_median(n,A_with_index)[1] # A의 median of median 을 찾는 과정
        small,same,big=partition(n,A,pivot_index) # pivot index를 기준으로 small, same, big element로 나눔
        if i<=len(small) : # 찾고자 하는 i가 small내에 존재
            return deterministic_selection(len(small),small,i)
        elif i>len(small)+len(same) : # 찾고자 하는 i가 big에 존재
            return deterministic_selection(len(big),big,i-len(small)-len(same))
        else : # 그외의 경우는 pivot_index 위치의 원소와 값이 같으므로 same을 타겟팅
            return same[0]

def write(result, t, i, name):
    f = open(name+str(i), 'w') # main.py가 있는 경로에 name, i 를 이용해 txt 파일 생성 및 오픈
    f.write(result) # 결과값 작성
    f.write(t) # 결과값 작성
    f.close()

def random_total(n,A,i,x):
    """
    :param n: Length of list A
    :param A: list A
    :param i: i-th smallest element
    :param x: 결과값 저장하는 txt naming에 사용될 문자열
    :return: randomized selection을 통해 나온 list A의 i-th smallest element
    """
    start = time.time() # 시간측정 시작
    r = randomized_selection(n, A, i) #randomized selection 실행
    end = time.time() # 시간측정 종료
    result = str(r) + "\n" # i-th smallest element 표시
    t = str(round((end - start) * (10 ** 3), 2)) + "ms" # 걸린 시간 ms 단위로 변환
    write(result, t, x, "random") # 결과값, 걸린 시간을 random+x 의 naming으로 기록
    return r

def deter_total(n,A,i,x):
    """
    :param n: Length of list A
    :param A: list A
    :param i: i-th smallest element
    :param x: 결과값 저장하는 txt naming에 사용될 문자열
    :return: deterministic selection을 통해 나온 list A의 i-th smallest element
    """
    start = time.time() # 시간측정 시작
    r = deterministic_selection(n, A, i) # deterministic selection 실행
    end = time.time() # 시간측정 종료
    result = str(r) + "\n" # i-th smallest element 표시
    t = str(round((end - start) * (10 ** 3), 2)) + "ms" # 걸린 시간 ms 단위로 변환
    write(result, t, x, "deter") # 결과값, 걸린 시간을 deter+x 의 naming으로 기록
    return r

def input_to_output(x):
    """
    읽는 input의 txt file name이
    ex. input.txt, input1.txt, input2.txt 이런 형식이므로
    ex. .txt, 1.txt, 2.txt 이렇게 리턴하여
    결과값 저장에 앞에 random, deter, result만 붙여서
    ex. random.txt, deter.txt, result.txt 이런식으로 저장하기 위함
    :param x: Ex. input.txt, input1.txt, input2.txt
    :return: Ex. .txt, 1.txt, 2.txt
    """
    x=x[5:]
    return (x)

def check(n,A,i,random,deter,output_file_name):
    """
    :param n: list A의 길이
    :param A: n개의 elements로 이루어진 list
    :param i: list A에서 i-th 작은 원소
    :param random: 위 n,A,i에 대해 randomized-selection의 결과로 나온 i-th smallest number
    :param deter: 위 n,A,i에 대해 deterministic-selection의 결과로 나온 i-th smallest number
    :param output_file_name: checker program 의 결과를 txt file에 저장할 때, txt file naming에 쓰일 문자열
    """
    small_r, big_r = 0, 0 # list 내에서 random 값보다 작은, 큰 elements의 수
    small_d, big_d = 0, 0 # list 내에서 deter 값보다 작은, 큰 elements의 수

    for j in range(n): # list A를 돌며 small_r, big_r, small_d, bid_d check
        if A[j]<random:
            small_r +=1
        elif A[j]>random:
            big_r += 1
        if A[j]<deter:
            small_d +=1
        elif A[j]>deter:
            big_d += 1

    if small_r<i and i<=n-big_r : # random의 결과를 확인하는 작업
        check_random = "randomized selection check : Correct\n"
    else : check_random = "randomized selection check : Wrong\n"
    if small_d<i and i<=n-big_d : # deter의 결과를 확인하는 작업
        check_deter = "deterministic selection check : Correct"
    else : check_deter = "deterministic selection check : Wrong"

    write(check_random,check_deter,output_file_name,"result") # txt file에 결과를 기록

# main
def main():
    input_list=glob.glob("input*.txt") # hw1 directory에서 'input*.txt' 형식의 파일들을 불러옴
    for input in input_list:
        n,A,i=read(input) # input text file에 대해 list의 길이, list, 몇번째 원소를 골라낼 것인가를 저장
        if n!=0 and A!=0 and i!=0:
            output_file_name=input_to_output(input) # input1.txt 파일이면 1.txt만 뽑아내서 프로그램 결과 txt 파일 naming에 사용
            random_result = random_total(n,A,i,output_file_name) # input에 대해 randomized selection 결과 저장
            deter_result = deter_total(n,A,i,output_file_name) # input에 대해 deterministic selection 결과 저장
            check(n,A,i,random_result,deter_result,output_file_name) # input과 위 random, deter selection 결과를 가지고, 맞는 결과인지 테스트

main()

