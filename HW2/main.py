import glob

def read(x):
    """
    x 를 open하여 읽어옴
    각 insturction이 'I 10' 이렇게 되어있던 것을 ['I',10(int)] 로 바꿈
    """
    try :
        f = open(x,'r')
        lines = f.readlines()
        f.close()
        a = []
        for i in range(len(lines)):
            instruction = []  # 안쪽 리스트로 사용할 빈 리스트 생성
            k=lines[i].strip('\n')
            k=k.split()
            instruction.append(k[0])
            instruction.append(int(k[1]))
             # 안쪽 리스트에 0 추가
            a.append(instruction)  # 전체 리스트에 안쪽 리스트를 추가
    except :
        a=0
    return a


class Node :
    """
    OS-Tree에서 각 노드에 해당하는 작업
    Node(data)로 선언하며 data가 0이 아니라면 해당 data를 갖고 자식을 Node(0)으로 갖는 노드 생성
    Red-Black Tree에서 NIL에 해당하는 node는 Node(0)으로 선언
    """
    def __init__(self, data) :
        if data != 0 :
            self.data = data
            self.parent = None
            self.l_child = Node(0)
            self.r_child = Node(0)
            self.rank = 0
            self.color = "red"
        else : # data가 0이면 NIL : red-black tree에서 leaf node로 사용되는 녀석
            self.data = 0
            self.parent = None
            self.l_child = None
            self.r_child = None
            self.rank = 0
            self.color = "black"


class OS_tree :

    def __init__(self):
        # 초기 OS_tree 선언 시 root는 Node(0)으로 기분 세팅
        self.root = Node(0)

    def left_rotate(self, x):
        # Red-Black Tree에서 left rotate 구현
        y=x.r_child
        p=x.parent
        dummy = y.rank # rank 바꿔주는 작업
        y.rank= x.rank

        # RL 손자 노드가 옮겨 붙는 것이므로 NIL인지 아닌지 나눠서 진행
        if y.l_child.data != 0 :
            y.l_child.parent=x
            x.rank = x.rank - dummy + y.l_child.rank
        else:
            x.rank = x.rank - dummy
        x.r_child=y.l_child
        x.parent=y
        y.l_child=x

        # y의 parent 지정 시 x가 root였는지 아닌지 나눠서 진행
        if x==self.root:
            self.root=y
            y.parent = None
        else :
            if p.l_child == x :
                p.l_child = y
                y.parent = p
            elif p.r_child == x:
                p.r_child = y
                y.parent = p

    def right_rotate(self, y):
        # Red-Black Tree에서 right rotate 구현
        x= y.l_child
        dummy = x.rank # rank 바꿔주는 작업
        x.rank = y.rank
        p = y.parent

        # LR 손자 노드가 옮겨 붙는 것이므로 NIL인지 아닌지 나눠서 진행
        if x.r_child.data != 0:
            x.r_child.parent = y
            y.rank = y.rank-dummy+x.r_child.rank
        else :
            y.rank = y.rank-dummy

        y.l_child = x.r_child

        y.parent = x
        x.r_child = y
        # X,Y의 parent 처리
        if y == self.root:
            self.root = x
            x.parent = None
        else:
            if p.l_child == y:
                p.l_child = x
                x.parent = p
            elif p.r_child == y:
                p.r_child = x
                x.parent = p

    def insert_value(self, target, parent, data):
        # Insert시 값을 알맞은 leaf node에 삽입
        if self.root.data==0 :
            # 첫 insert의 경우
            target = Node(data)
            target.rank=1
            self.root = target
            # root color는 black 이므로
            self.root.color = "black"
            self.check_color_insert(target)
            return data

        elif target.data == 0 :
            # data가 들어가야 할 Node(0)을 찾은 경우
            target = Node(data)
            target.parent = parent
            target.rank=1
            # parent와의 연결 작업
            if parent.data>data :
                parent.l_child = target
            elif parent.data<data :
                parent.r_child = target
            # 이 Node를 subtree의 구성요소로 포함하고 있는 node들의 rank +1
            self.rank_add_after_insert(target)
            # Red-Black tree의 color 체크
            self.check_color_insert(target)

            return target.data

        else :
            # 알맞은 insert 위치를 찾아가는 과정
            if target.data>data :
                return self.insert_value(target.l_child, target, data)
            elif target.data<data:
                return self.insert_value(target.r_child, target, data)
            else :
                # 해당 데이터가 이미 입력된 경우
                return 0

    def rank_add_after_insert(self, me):
        # insert되면서 거쳐온 node들 rank를 1씩 증가시키는 작업
        if me.parent is not None :
            me.parent.rank +=1
            self.rank_add_after_insert(me.parent)

    def check_color_insert(self, me):
        if me.parent is None :
            # root로 입력된 경우
            me.color="black"
        else :
            # parent가 black이면 삽입 노드 red로 하면 상관없음
            # parent가 red인 경우만 살피기
            if me.parent.color == "red":
                # parent의 다른 형제노드를 s라 칭하는 작업
                if me.parent == me.parent.parent.l_child:
                    s=me.parent.parent.r_child
                elif me.parent == me.parent.parent.r_child:
                    s=me.parent.parent.l_child

                # s가 red인 경우
                if s.data!=0 and s.color=="red":
                    me.parent.color = "black"
                    s.color = "black"
                    me.parent.parent.color="red"
                    self.check_color_insert(me.parent.parent)
                else : # s가 black인 경우
                    # pp의 LR 손자 노드인 경우
                    if me == me.parent.r_child and me.parent == me.parent.parent.l_child:
                        p=me.parent
                        pp=p.parent
                        self.left_rotate(p)
                        self.right_rotate(pp)
                        me.color = "black"
                        pp.color = "red"
                    # pp의 LL 손자 노드인 경우
                    elif me == me.parent.l_child and me.parent == me.parent.parent.l_child:
                        self.right_rotate(me.parent.parent)
                        me.parent.color = "black"
                        me.parent.r_child.color = "red"
                    # pp의 RR 손자 노드인 경우
                    elif me == me.parent.r_child and me.parent == me.parent.parent.r_child :
                        self.left_rotate(me.parent.parent)
                        me.parent.color = "black"
                        me.parent.l_child.color = "red"
                    # pp의 RL 손자 노드인 경우
                    elif me == me.parent.l_child and me.parent == me.parent.parent.r_child :
                        p = me.parent
                        pp = p.parent
                        self.right_rotate(p)
                        self.left_rotate(pp)
                        me.color = "black"
                        pp.color = "red"


    def find_delete_target(self, target_node, x):
        """
        x를 갖는 타겟 노드를 찾는다.
        발견하지 못하면 return 0
        발견하는 경우
        1.two children nodes -> successor 찾고 data 교환, return successor node
        2.one or no child node -> return target node
        """
        if  target_node.data == 0:
            return 0
        elif target_node.data < x :
            return self.find_delete_target(target_node.r_child, x)
        elif target_node.data>x :
            return self.find_delete_target(target_node.l_child, x)
        elif target_node.data == x :
            result = x # 리턴하기 위함
            if target_node.r_child.data!=0 and target_node.l_child.data!=0 :
                # two children node를 가지는 경우
                # min successor를 찾는다
                r_subtree_root_node = target_node.r_child
                while (r_subtree_root_node.l_child.data != 0):
                    r_subtree_root_node = r_subtree_root_node.l_child
                successor_node = r_subtree_root_node
                target_node.data, successor_node.data = successor_node.data, target_node.data
                target_node = successor_node

            self.rank_minus_find_delete_target(target_node) # 지워질 노드를 subtree의 노드로 갖는 노드들 rank를 각각 -1
            self.delete_node(target_node) # 해당 node 지우고 color check까지 진행
            return result

    def rank_minus_find_delete_target(self, me):
        # delete할 node의 조상 node rank를 1씩 감소시키는 작업
        if me.parent is not None :
            me.parent.rank -=1
            self.rank_minus_find_delete_target(me.parent)

    def delete_node(self, target):
        # target은 현재 0 or 1 child node를 가짐
        m = target
        if target == self.root : # 타겟이 root인 경우
            if target.l_child.data == 0 :
                self.root = target.r_child
            else :
                self.root = target.l_child
            self.root.color = "black"
            return

        if target.parent.l_child == target : # case 1 : p의 L child 인 경우
            # target의 parent와 child를 직접 연결시켜주는 작업
            if target.r_child.data!=0 :
                target.parent.l_child = target.r_child
                target.r_child.parent = target.parent
            else :
                target.parent.l_child = target.l_child
                target.l_child.parent = target.parent
            if target.color == "black" :
                # 지운 노드가 red면 상관없음 : 새로 이어준 parent와 child가 모두 black이었기 때문
                self.check_color_delete(target.parent.l_child, 1) # case 1에 대해 color check

        elif target.parent.r_child == target : # case 2 : p의 R child 인 경우
            # target의 parent와 child를 직접 연결시켜주는 작업
            if target.r_child.data!=0 :
                target.parent.r_child = target.r_child
                target.r_child.parent = target.parent
            else :
                target.parent.r_child = target.l_child
                target.l_child.parent = target.parent
            if target.color == "black":
                # 지운 노드가 red면 상관없음 : 새로 이어준 parent와 child가 모두 black이었기 때문
                self.check_color_delete(target.parent.r_child, 2) # case 2에 대해 color check


    def check_color_delete(self, x ,case):
        """
        parent, child를 연결한 후(= 해당 노드를 지운 후) 지운 노드가 black이면 color에 문제가 생김
        color check를 통해 red-black tree rule이 지켜지도록 함
        """
        if case == 1 :
            if x==self.root : # child가 root로 새로 올라온 경우
                x.color = "black"
                return
            p = x.parent # 아닌경우 parent를 찾을 수 있음
            if x.color == "red": #  black이 지워졌으므로, red -> black으로 바꾸면 끝
                x.color = "black"
            elif x.color == "black" :
                s = p.r_child # parent의 다른 자식(s)을 가지고 아래의 경우들로 나눠서 진행
                if s.color == "red" : # s가 red인 경우, p 중심으로 left_rotate 후 x에 대해 다시 체크
                    p.color = "red"
                    s.color = "black"
                    self.left_rotate(p)
                    self.check_color_delete(x,1)
                elif s.color == "black" and s.l_child.color == "black" and s.r_child.color == "black" :
                    # s가 black이고 s의 두 자식이 black
                    s.color = "red"
                    if p.color=="red":
                        p.color="black"
                    elif p.color == "black":
                        if p == self.root :
                            return
                        else :
                            if p == p.parent.l_child:
                                self.check_color_delete(p,1)
                            else :
                                self.check_color_delete(p,2)
                elif s.color == "black" and s.l_child.color == "red" and s.r_child.color == "black" :
                    # s가 black이고 나와 가까운 s의 자식만 red
                    s.color = "red"
                    s.l_child.color = "black"
                    self.right_rotate(s)
                    self.check_color_delete(x,1)
                elif s.color == "black" and s.r_child.color == "red" :
                    # s가 black이고 나와 먼 s의 자식이 red
                    s.color = p.color
                    p.color = "black"
                    s.r_child.color = "black"
                    self.left_rotate(p)

        if case == 2 :
            # 위의 case 1에 대해 좌우 대칭을 한 경우들
            # 좌우 대칭 고려하여 유사하게 작업하면 됨
            if x==self.root :
                x.color = "black"
                return
            p = x.parent
            if x.color == "red":
                x.color = "black"
            elif x.color == "black" :
                s = p.l_child
                if s.color == "red" :
                    p.color = "red"
                    s.color = "black"
                    self.right_rotate(p)
                    self.check_color_delete(x,2)
                elif s.color == "black" and s.l_child.color == "black" and s.r_child.color == "black" :
                    s.color = "red"
                    if p.color=="red":
                        p.color="black"
                    elif p.color == "black":
                        if p == self.root :
                            return
                        else :
                            if p == p.parent.l_child:
                                self.check_color_delete(p,1)
                            else :
                                self.check_color_delete(p,2)
                elif s.color == "black" and s.r_child.color == "red" and s.l_child.color == "black" :
                    s.color = "red"
                    s.r_child.color = "black"
                    self.left_rotate(s)
                    self.check_color_delete(x,2)
                elif s.color == "black" and s.l_child.color == "red" :
                    s.color = p.color
                    p.color = "black"
                    s.l_child.color = "black"
                    self.right_rotate(p)

    def find_element(self, target, i):
        # i번째 element를 찾는 작업
        if target.data == 0 :
            # 찾지못하고 leaf node인 Node(0)에 도달
            return 0
        else :
            r = target.l_child.rank + 1 # 지금 target Node의 data가 몇번째인지 알려줌
            if r == i : # 같으면 찾는 원소가 target의 data
                return target.data
            elif r < i : # 더 큰 데이터를 찾는 중이므로 r_child로 이동해서 진행
                return self.find_element(target.r_child, i-r)
            elif i < r : # 더 작은 데이터를 찾는 중이므로 l_child로 이동해서 진행
                return self.find_element(target.l_child, i)

    def find_node(self, target, x):
        # x를 갖는 node를 찾아줌
        if target.data == 0 :
            # 못 찾으면 Node(0) return
            return target
        elif target.data == x :
            # 같은 데이터 찾으면 해당 노드 리턴
            return target
        elif target.data > x :
            # 더 작은 데이터를 찾으므로 l_child로 이동 후 진행
            return self.find_node(target.l_child, x)
        elif target.data < x :
            # 더 큰 데이터를 찾으므로 r_child로 이동 후 진행
            return self.find_node(target.r_child, x)

    def find_rank(self, x):
        target = self.find_node(self.root, x)
        # x를 가지는 node를 target에 받아옴
        if target.data == 0 :
            # x를 Tree가 갖지않음
             return 0
        else :
            # 내가 r_child라면 l_child가 root인 subtree와 parent node의 데이터들보다는 큼
            # r_child인 경우만 l_child가 root인 subtree 원소 수와 parent node까지 내 밑의 숫자 갯수에 추가
            # root에 도달하게 되면 종료
            r = target.l_child.rank + 1
            y = target
            while y != self.root :
                if y == y.parent.r_child :
                    r = r + y.parent.l_child.rank + 1
                y = y.parent
            return r


def OS_Insert(T, x):
    # T라는 OS-Tree에 x를 insert
    # 성공하면 x, 중복인 경우 0 리턴
    value = T.insert_value(T.root, None, x)
    return value

def OS_Delete(T,x):
    # T라는 OS-Tree에 x를 delete
    # delete에 성공하면 x, 지우는 원소가 없으면 0 리턴
     value = T.find_delete_target(T.root, x)
     return value

def OS_Select(T,x):
    # T라는 OS-Tree에 x-th element 를 보여줌
    # 찾아내면 그 원소의 값 리턴, 범위를 넘어서면 0 리턴
    value = T.find_element(T.root, x)
    return value

def OS_Rank(T,x):
    # T라는 OS-Tree에서 원소 x가 몇번째인지 알려줌
    # 원소 x가 있으면 몇번째인지 리턴, x가 Tree에 없다면 0 리턴
    value = T.find_rank(x)
    return value

def write(filename, result):
    # input.txt파일 오픈
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    # output txt파일 이름 설정을 위한 작업
    s1 = filename[-9:]
    s2 = s1.replace('input','output')
    new_filename = filename[:-9]+s2
    # input 내용 옮겨 적는 작업
    final_str = ''
    for i in lines:
        final_str=final_str+str(i)
    # output txt 결과값 작성
    fw = open(new_filename, 'w') # txt 파일 생성 및 오픈
    fw.write(final_str+result) # 결과값 작성
    fw.close()

def compare(input,output):
    # input, output의 본질적인 checker 프로그램 역할
    l=len(input)
    # 9999개의 길이를 갖고 각 요소가 0인 A set
    A=[]
    for i in range(9999):
        A.append(0)
    result = 'True'
    for i in range(l):
        # 각 instruction 별로 체크
        k = input[i].strip('\n')
        k = k.split()
        r = int(output[i].strip('\n'))
        oper = k[0] # insert, delete, select, rank의 앞글자를 따서 입력
        n = int(k[1]) # 지시사항에 해당하는 원소
        if oper =='I': # insert의 경우
            if A[n-1]==0 : # 원소가 tree에 없는 경우
                A[n-1] += 1
                if r != n :
                    result = 'False'
                    return result
            elif A[n-1]==1 : # 원소가 tree에 있는 경우
                if r != 0:
                    result = 'False'
                    return result

        elif oper=='D': # delete의 경우
            if A[n-1]==1 : # 지울 대상이 tree에 있는 경우
                A[n-1] -= 1
                if r != n :
                    result = 'False'
                    return result
            elif A[n-1]==0 : # 지울 대상이 tree에 없는 경우
                if r != 0:
                    result = 'False'
                    return result

        elif oper=='S': # select의 경우
            j=1
            if n==0 :
                return 'False'

            while (sum(A[0:j])!=n and j<10000) :
                j+=1

            if j< 10000 : # n-th번째 원소를 찾아낸 경우
                if r != j:
                    result = 'False'
                    return result
            elif j >= 10000 : # n-th번째 원소를 못 찾아낸 경우
                if r != 0 :
                    result = 'False'
                    return result

        elif oper=='R': # rank의 경우
            if n>0 and n<10000 and A[n-1]!=0: # 해당 원소가 Tree에 존재하는 경우
                if sum(A[:n]) != r :
                    result = 'False'
                    return result
            else :
                if r != 0 :
                    result = 'False'
                    return result
        else :
            return 'False'

    return result

def checker(input, output):
    # 배열 형태로 input, output txt파일의 내용을 저장하고
    # output.txt에서 input.txt와 겹치는 내용 제외해서 compare 함수에 보냄
    # compare 결과가 문자열 True, False 중 하나이므로
    # 이를 받아서 checker.txt에 write
    f = open(input, 'r')
    input_lines = f.readlines()
    f.close()
    f = open(output, 'r')
    output_lines = f.readlines()
    f.close()
    input_l = len(input_lines)
    output_l = len(output_lines)
    i = input_lines
    o = output_lines[input_l:]
    if input_l * 2 == output_l :
        result = compare(i,o)
    else : result = 'False'

    s1 = input[-9:]
    s2 = s1.replace('input', 'checker')
    new_filename = input[:-9] + s2

    fw = open(new_filename, 'w')  # txt 파일 생성 및 오픈
    fw.write(result)  # 결과값 작성
    fw.close()



def main():
    input_list=glob.glob("input*/input.txt") # hw2 directory에서 'input*.txt' 형식의 파일들을 불러옴
    for input in input_list:
        instruction= read(input) # instruction은 2차원 리스트로 각 element는 ['I','6'] 같은 형식, l은 길이
        l=len(instruction)
        T=OS_tree() # empty OS Tree 선언
        s1 = input[-9:]
        s2 = s1.replace('input', 'output')
        output = input[:-9] + s2
        result = ""
        for i in range(l):
            # 각 instruction별로 작업을 나눠서 진행하며 해당 명령어 결과값을 result에 줄 바꿔가며 저장
            if instruction[i][0]=='I':
                result = result + '\n' + str(OS_Insert(T,int(instruction[i][1])))
            elif instruction[i][0]=='D':
                result = result + '\n' + str(OS_Delete(T,int(instruction[i][1])))
            elif instruction[i][0] == 'S':
                result = result + '\n' + str(OS_Select(T,int(instruction[i][1])))
            elif instruction[i][0] == 'R':
                result = result + '\n' + str(OS_Rank(T,int(instruction[i][1])))
        result = result.strip('\n')
        write(input, result) # 해당 결과값을 이용하여 output.txt를 작성
        checker(input, output) # input, output을 이용해 checker.txt를 작성

main()




