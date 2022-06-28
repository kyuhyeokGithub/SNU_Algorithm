import math
import random
import sys
import time
from operator import itemgetter
from random import randint
sys.setrecursionlimit(10 ** 6)

class list_Node:
    """
    adjacency list를 통해 그래프를 표현할 때 Linked List의 각 노드가 되는 부분
    adjacency Array에서는 각 요소가 되며 value은 노드가 가리키는 Array의 size
    """
    def __init__(self, x):
        self.value = x
        self.next = None

class Graph_adj_list:
    # adj_list는 adjacency list에서의 리스트, 각 리스트 한칸은 Linked List를 가짐
    def __init__(self,vertex_num):
        self.vertex_num = vertex_num
        self.graph = []
        self.visited = []

        for i in range(self.vertex_num):
            self.graph.append(list_Node(0))
            self.visited.append(0)

    def insert(self,node_from,node_to):
        # adj_list[node_from-1]은 Linked List를 가지고
        # Linked List의 각 요소의 value값이 node_to 가 됨
        new_node = list_Node(node_to)
        target_vertex = self.graph[node_from-1]

        target_node = target_vertex
        while( target_node.next is not None ):
            if( target_node.next.value > node_to ) :
                break
            target_node = target_node.next
        new_node.next = target_node.next
        target_node.next = new_node

    def transpose(self):
        "node_to, from 을 바꾸어서 리턴"
        L_t = Graph_adj_list(self.vertex_num)

        for i in range(self.vertex_num):
            temp = self.graph[i]
            temp = temp.next
            while temp:
                L_t.insert(temp.value,i+1)
                temp = temp.next

        return L_t

    def get_list(self, node_from):
        """
        node_from + 1 의 노드의 근접 노드를 리스트로 만들어 반환
        :param node_from: i라면 node i+1 을 의미, 기준점 노드
        :return: node_from 에서 근접한 노드 리스트, visited 여부 상관x
        """
        adj = []
        temp = self.graph[node_from]
        temp = temp.next
        while temp:
            adj.append(temp.value-1)
            temp = temp.next
        return adj

    def visit(self, visit_node):
        "node를 방문했다고 표시"
        self.visited[visit_node-1] = 1

class Graph_adj_matrix:
    def __init__(self, vertex_num):
        self.vertex_num = vertex_num # 총 몇개의 vertex가 있는지
        self.graph = [] # i,j element는 i+1 node에서 j+1 node로의 직접적인 edge가 있는지 표시 ( 있으면 1, 아니면 0 )
        self.visited = [] # 각 Node에 대해 방문했는지 여부를 나타냄
        for i in range(vertex_num):
            self.graph.append([0 for j in range(vertex_num)])
            self.visited.append(0)

    def insert(self, node_from, node_to):
        # node_from에서 node_to 로의 Edge가 존재함
        self.graph[node_from-1][node_to-1]=1

    def visit(self, visit_node):
        "node를 방문했다고 표시"
        self.visited[visit_node-1] = 1

    def get_list(self, node_from):
        """
        node_from + 1 의 노드의 근접 노드를 리스트로 만들어 반환
        :param node_from: i라면 node i+1 을 의미, 기준점 노드
        :return: node_from 에서 근접한 노드 리스트, visited 여부 상관x
        """
        adj = []
        temp = self.graph[node_from]
        for i in range(self.vertex_num):
            if temp[i]==1:
                adj.append(i)

        return adj

    def transpose(self):
        "class의 adj_matrix 를 transpose 하여 Graph_adj_matrix class를 리턴"
        M_t = Graph_adj_matrix(self.vertex_num)
        for i in range(self.vertex_num):
            for j in range(self.vertex_num):
                M_t.graph[j][i]=self.graph[i][j]
        return M_t

class Graph_adj_array:
    def __init__(self, vertex_num):
        self.vertex_num = vertex_num # 총 몇개의 vertex가 있는지
        self.graph = [] # list_Node의 list로 각 list_Node의 value는 adjacent vertex 갯수, next는 adjacent node list
        self.visited = [] # 각 Node에 대해 방문했는지 여부를 나타냄
        for i in range(vertex_num):
            self.graph.append(list_Node(0))
            self.visited.append(0)
            self.graph[i].next = []

    def insert(self, node_from, node_to):
        # node_from에서 node_to 로의 Edge가 존재함
        target_array = self.graph[node_from-1].next
        l = self.graph[node_from-1].value
        if  l == 0 :
            # adjacent node가 비어있는 경우
            target_array.append(node_to)
            self.graph[node_from-1].value+=1
        else :
            # adjacent node가 비어있지 않은 경우
            # node_to를 정렬하여 삽입
            i=0
            while (i<l):
                if target_array[i]>node_to :
                    break
                i+=1
            target_array.insert(i, node_to)
            self.graph[node_from-1].value += 1

    def visit(self, visit_node):
        "node를 방문했다고 표시"
        self.visited[visit_node-1] = 1

    def get_list(self, node_from):
        """
        node_from + 1 의 노드의 근접 노드를 리스트로 만들어 반환
        :param node_from: i라면 node i+1 을 의미, 기준점 노드
        :return: node_from 에서 근접한 노드 리스트, visited 여부 상관x
        """
        target = self.graph[node_from].next
        adj = []
        for i in target:
            adj.append(i-1)
        return adj

    def transpose(self):
        "class의 adj_array 를 transpose 하여 Graph_adj_array class를 리턴"
        A_t = Graph_adj_array(self.vertex_num)
        for i in range(self.vertex_num):
            for j in range(self.graph[i].value):
                A_t.insert(self.graph[i].next[j], i+1)
        return A_t

def start_with_order(G ,home_node, order):
    """
    type 상관없이 G에서 home_node를 기준으로 인접 node를 방문하고
    더이상 갈 수 없다면 order에 추가
    :param G: matrix, list, array 상관없이 그래프를 표현하는 class
    :param home_node: 기준점 노드 -1에 해당하는 값
    :param order: 방문시 finish 순서에 따른 노드 리스트
    :return: 리턴값은 없음
    """
    G.visited[home_node] = 1
    adj_node = G.get_list(home_node)
    if not adj_node:
        order.append(home_node)
    else :
        for i in adj_node :
            if G.visited[i]==0 :
                start_with_order(G, i, order)
        order.append(home_node)

def start(G ,home_node, SCC):
    """
    G에서 home_node를 기준으로 인접 node를 방문하고
    더이상 갈 수 없다면 멈추고
    그간 SCC에 쌓인 노드들끼리 SCC 구성
    :param G: matrix, list, array 상관없이 그래프를 표현하는 class
    :param home_node: 기준점 노드 -1에 해당하는 값
    :param SCC: home_node와 SCC 관계에 있는 노드들 리스트로, 그간 거쳐온 노드들 중 SCC이면 리스트에 원소로 있음
    :return: 리턴값은 없음
    """
    G.visited[home_node] = 1
    SCC.append(home_node+1)
    adj_node = G.get_list(home_node)

    if adj_node :
        for i in adj_node :
            if G.visited[i]==0 :
                start(G, i, SCC)



def first_DFS(G):
    """
    SCC-algorithm에서 첫번째 DFS 작업
    each vertex에 대해 order 리스트에 finish 순서로 노드들을 기록
    """
    vertex_num = G.vertex_num
    visited = G.visited
    order = []
    for i in range(vertex_num):
        if visited[i]==0:
            start_with_order(G, i, order)
    return order

def second_DFS(G, order):
    """
    order의 역순으로 G의 노드들을 방문하며
    start 함수를 이용해, 더이상 방문하지 않은 근접 노드가 없을 때
    sub_SCC에 쌓인 노드들 리스트끼리 SCC 관계를 이룸
    다시 방문하지 않은 노드들에 대해 반복 작업 수행
    """
    vertex_num = G.vertex_num
    visited = G.visited
    result = []


    for i in range(vertex_num-1,-1,-1):
        if visited[order[i]] == 0:
            sub_SCC = []
            start(G, order[i], sub_SCC)
            result.append(sub_SCC)

    return result



def find_SCC(G):
    """
    1. first DFS : Run DFS on G to compute finish time for each vertex
    2. Compute G_t (transpose) where direction of each edge in G is reversed
    3. Run DFS on G_t with visiting vertices in decreasing order of finish time
    """
    order = first_DFS(G)
    G_t = G.transpose()
    SCC_list = second_DFS(G_t, order)
    return SCC_list

def read_txt(x):
    """
    x 경로를 가지는 input 파일을 읽고
    라인별로 저장한 lines 이라는 list 를 리턴
    """
    try :
        f = open(x,'r')
        lines = f.readlines()
        f.close()
        return lines
    except :
        return lines

def Insert_data(G, lines, n):
    """
    읽어드린 lines 에 대해
    graph에 필요한 vertex num을 설정하고
    상황에 맞게 edge들을 G에 기록하는 작업
    """
    for i in range(1,n+1,1):
        node_list = lines[i].strip('\n').split(" ")
        for j in node_list:
            if int(j) == 0:
                break
            else :
                G.insert(i,int(j))


def write_txt(file_path, SCC_list, SCC_time):
    """
    주어진 file_path 에 대해 output txt 파일을 만들고
    SCC list와 SCC time을 기록
    """
    fw = open(file_path, 'w')  # txt 파일 생성 및 오픈
    for i in SCC_list:
        fw.write(" ".join(i))
        fw.write("\n")
    fw.write(SCC_time)
    fw.close()


def main():
    input_lines = read_txt(sys.argv[1]) # input txt 파일의 경로를 설정
    vertex_num = int(input_lines[0]) # Graph 에 필요한 vertices num 설정
    # 지정한 type에 따라 G의 type 지정하여 생성
    if sys.argv[3]=="adj_list":
        G = Graph_adj_list(vertex_num)
    elif sys.argv[3]=="adj_mat":
        G = Graph_adj_matrix(vertex_num)
    elif sys.argv[3]=="adj_arr":
        G = Graph_adj_array(vertex_num)
    Insert_data(G,input_lines, vertex_num) # G의 edge들을 기록

    start = time.time()  # 시간측정 시작
    SCC_list = find_SCC(G)
    end = time.time()  # 시간측정 종료
    SCC_time = str(round((end-start)*(10**3),2)) + "ms"  # 걸린 시간 ms 단위로 변환

    # SCC_list 정렬 후, 중복 노드 없으므로 다시 첫 node로 정렬
    for i in range(len(SCC_list)):
        SCC_list[i].sort()

        for j in range(len(SCC_list[i])):
            SCC_list[i][j]=str(SCC_list[i][j])

    SCC_list.sort(key=itemgetter(0))

    write_txt(sys.argv[2], SCC_list, SCC_time) # 기록



def make_input(n,i):
    """
    n개의 vertex를 갖는 input txt 파일을 만들기 위한 함수
    """
    fw = open("input"+str(i)+"/input.txt", 'w')  # txt 파일 생성 및 오픈
    fw.write(str(n))
    fw.write("\n")
    cnt = int(math.sqrt(math.sqrt(n)))

    e = 300*n

    for i in range(n):

        if cnt == 0 :
            fw.write("0")
        else :
            if i!=n-1 :
                v = 2*int(e/(n-i))
                k = randint(0,v)
                e = e-k
                a = random.sample(range(1,n+1,1),k)

            else :
                a = random.sample(range(1,n+1,1), e)
            if (i+1) in a :
                a.remove(i+1)
            if a :
                fw.write(" ".join(str(e) for e in a))
            else :
                fw.write("0")
        if i!=n-1 :
            fw.write("\n")
    fw.close()

main()
