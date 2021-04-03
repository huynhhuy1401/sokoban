import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                if (newPosPlayer, newPosBox) not in exploredSet:
                    frontier.append(node + [(newPosPlayer, newPosBox)])
                    actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState): # Ham thuc hien BFS
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) # Vi tri bat dau cua cac thung 
    beginPlayer = PosOfPlayer(gameState) # Vi tri bat dau cua nguoi choi

    startState = (beginPlayer, beginBox) # Trang thai khoi dau vd: ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = collections.deque([[startState]]) # Luu tru trang thai khoi dau vao frontier
    actions = collections.deque([[0]]) # Luu tru hanh dong tuong ung voi trang thai khoi dau vao actions
    exploredSet = set() # exploredSet de chua cac trang thai da mo rong
    temp = [] # mang temp de luu cac hanh dong dan den goal
    ### Implement breadthFirstSearch here
    while frontier: # while frotiner chua trong
        node = frontier.popleft() # lay ra trang thai o vi tri dau tien trong frontier luu vao node
        node_action = actions.popleft() # lay ra hanh dong tuong ung voi trang thai duoc lay ra luu vao node_action
        if isEndState(node[-1][-1]): # neu day la trang thai dich (goal)
            temp += node_action[1:] # luu cac hanh dong dan den goal vao temp
            break # dung vong lap
        if node[-1] not in exploredSet: # Neu trang thai trong node chua duoc mo rong
            exploredSet.add(node[-1]) # them trang thai trong node vao exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Voi moi hanh dong tu cac hanh dong co the thuc hien duoc tu trang thai hien tai trong node
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Vi tri moi cua nguoi choi va cac thung dua vao hanh dong duoc thuc hien 
                if isFailed(newPosBox): # Neu vi tri cac thung ko the giai duoc 
                    continue # Qua hanh dong ke tiep trong cac hanh dong
                
                frontier.append(node + [(newPosPlayer, newPosBox)]) # Them vao frontier trang thai moi chua vi tri moi cua nguoi choi va cac thung
                actions.append(node_action + [action[-1]]) # Them vao actions chuoi hanh dong dua den trang thai moi duoc them vao frontier
    return temp # Tra ve mang luu cac hanh dong dan den goal
    
def cost(actions): # Ham cost cho UCS
    """A cost function"""
    return len([x for x in actions if x.islower()]) # Dem cac hanh dong la chu thuong roi tra ve ket qua da dem

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # Vi tri bat dau cua cac thung
    beginPlayer = PosOfPlayer(gameState) # Vi tri bat dau cua nguoi choi

    startState = (beginPlayer, beginBox) # Trang thai khoi dau vd: ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = PriorityQueue() # Khai bao frontier de luu cac trang thai
    frontier.push([startState], 0) # Dua vao frontier trang thai ban dau va cost cua no (bang 0)
    exploredSet = set() # exploredSet de chua cac trang thai da mo rong
    actions = PriorityQueue() # Khai bao actions 
    actions.push([0], 0) # Luu tru hanh dong tuong ung voi trang thai khoi dau vao actions va cost cua hanh dong
    temp = [] # mang temp de luu cac hanh dong dan den goal
    ### Implement uniform ((cost search here
    while not frontier.isEmpty(): # while frotiner chua trong
        node = frontier.pop() # lay ra trang thai co cost thap nhat
        node_action = actions.pop() # lay ra hanh dong tuong ung voi trang thai tren
        if isEndState(node[-1][-1]): # neu day la trang thai dich (goal)
            temp += node_action[1:] # luu cac hanh dong dan den goal vao temp
            break # dung vong lap
        if node[-1] not in exploredSet: # Neu trang thai trong node chua duoc mo rong
            exploredSet.add(node[-1]) # them trang thai trong node vao exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Voi moi hanh dong tu cac hanh dong co the thuc hien duoc tu trang thai hien tai trong node
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Vi tri moi cua nguoi choi va cac thung dua vao hanh dong duoc thuc hien 
                if isFailed(newPosBox): # Neu vi tri cac thung ko the giai duoc 
                    continue # Qua hanh dong ke tiep trong cac hanh dong
                newActions = node_action + [action[-1]] # Chuoi hanh dong moi
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(newActions[1:])) # Them vao frontier trang thai moi chua vi tri moi cua nguoi choi va cac thung va cost cua chuoi hanh dong tuong ung
                actions.push(node_action + [action[-1]], cost(newActions[1:])) # Them vao actions chuoi hanh dong moi va cost cua chuoi hanh dong tuong ung
    return temp # Tra ve mang luu cac hanh dong dan den goal

def calc_manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic1(posBox): # Khoang cach giua cac hop va target
    p1 = sorted(posBox)
    p2 = sorted(posGoals)
    total = 0
    for i in range(len(p1)):
        total += calc_manhattan(p1[i], p2[i])
    return total

def heuristic2(posPlayer, posBox): # Khoang cach gan nhat cua nguoi choi va cac hop
    distances = [calc_manhattan(posPlayer, posBox[i]) for i in range(len(posBox))]
    return min(distances)

def greedyFirstSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # Vi tri bat dau cua cac thung
    beginPlayer = PosOfPlayer(gameState) # Vi tri bat dau cua nguoi choi

    startState = (beginPlayer, beginBox) # Trang thai khoi dau vd: ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = PriorityQueue() # Khai bao frontier de luu cac trang thai
    frontier.push([startState], 0) # Dua vao frontier trang thai ban dau va cost cua no (bang 0)
    exploredSet = set() # exploredSet de chua cac trang thai da mo rong
    actions = PriorityQueue() # Khai bao actions 
    actions.push([0], 0) # Luu tru hanh dong tuong ung voi trang thai khoi dau vao actions va cost cua hanh dong
    temp = [] # mang temp de luu cac hanh dong dan den goal
    ### Implement uniform ((cost search here
    while not frontier.isEmpty(): # while frotiner chua trong
        node = frontier.pop() # lay ra trang thai co cost thap nhat
        node_action = actions.pop() # lay ra hanh dong tuong ung voi trang thai tren
        if isEndState(node[-1][-1]): # neu day la trang thai dich (goal)
            temp += node_action[1:] # luu cac hanh dong dan den goal vao temp
            break # dung vong lap
        if node[-1] not in exploredSet: # Neu trang thai trong node chua duoc mo rong
            exploredSet.add(node[-1]) # them trang thai trong node vao exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Voi moi hanh dong tu cac hanh dong co the thuc hien duoc tu trang thai hien tai trong node
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Vi tri moi cua nguoi choi va cac thung dua vao hanh dong duoc thuc hien 
                if isFailed(newPosBox): # Neu vi tri cac thung ko the giai duoc 
                    continue # Qua hanh dong ke tiep trong cac hanh dong
                newActions = node_action + [action[-1]] # Chuoi hanh dong moi
                #frontier.push(node + [(newPosPlayer, newPosBox)], cost(newActions[1:]) + heuristic1(newPosBox)) # Them vao frontier trang thai moi chua vi tri moi cua nguoi choi va cac thung va cost + heuristic1 cua chuoi hanh dong tuong ung
                #actions.push(node_action + [action[-1]], cost(newActions[1:]) + heuristic1(newPosBox)) # Them vao actions chuoi hanh dong moi va cost + heuristic1 cua chuoi hanh dong tuong ung
                frontier.push(node + [(newPosPlayer, newPosBox)], heuristic2(newPosPlayer, newPosBox)) # Them vao frontier trang thai moi chua vi tri moi cua nguoi choi va cac thung va cost + heuristic1 cua chuoi hanh dong tuong ung
                actions.push(node_action + [action[-1]], heuristic2(newPosPlayer, newPosBox)) # Them vao actions chuoi hanh dong moi va cost + heuristic1 cua chuoi hanh dong tuong ung
    return temp # Tra ve mang luu cac hanh dong dan den goal

def aStarSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) # Vi tri bat dau cua cac thung
    beginPlayer = PosOfPlayer(gameState) # Vi tri bat dau cua nguoi choi

    startState = (beginPlayer, beginBox) # Trang thai khoi dau vd: ((2, 2), ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5)))
    frontier = PriorityQueue() # Khai bao frontier de luu cac trang thai
    frontier.push([startState], 0) # Dua vao frontier trang thai ban dau va cost cua no (bang 0)
    exploredSet = set() # exploredSet de chua cac trang thai da mo rong
    actions = PriorityQueue() # Khai bao actions 
    actions.push([0], 0) # Luu tru hanh dong tuong ung voi trang thai khoi dau vao actions va cost cua hanh dong
    temp = [] # mang temp de luu cac hanh dong dan den goal
    ### Implement uniform ((cost search here
    while not frontier.isEmpty(): # while frotiner chua trong
        node = frontier.pop() # lay ra trang thai co cost thap nhat
        node_action = actions.pop() # lay ra hanh dong tuong ung voi trang thai tren
        if isEndState(node[-1][-1]): # neu day la trang thai dich (goal)
            temp += node_action[1:] # luu cac hanh dong dan den goal vao temp
            break # dung vong lap
        if node[-1] not in exploredSet: # Neu trang thai trong node chua duoc mo rong
            exploredSet.add(node[-1]) # them trang thai trong node vao exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): # Voi moi hanh dong tu cac hanh dong co the thuc hien duoc tu trang thai hien tai trong node
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # Vi tri moi cua nguoi choi va cac thung dua vao hanh dong duoc thuc hien 
                if isFailed(newPosBox): # Neu vi tri cac thung ko the giai duoc 
                    continue # Qua hanh dong ke tiep trong cac hanh dong
                newActions = node_action + [action[-1]] # Chuoi hanh dong moi
                #frontier.push(node + [(newPosPlayer, newPosBox)], cost(newActions[1:]) + heuristic1(newPosBox)) # Them vao frontier trang thai moi chua vi tri moi cua nguoi choi va cac thung va cost + heuristic1 cua chuoi hanh dong tuong ung
                #actions.push(node_action + [action[-1]], cost(newActions[1:]) + heuristic1(newPosBox)) # Them vao actions chuoi hanh dong moi va cost + heuristic1 cua chuoi hanh dong tuong ung
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(newActions[1:]) + heuristic2(newPosPlayer, newPosBox)) # Them vao frontier trang thai moi chua vi tri moi cua nguoi choi va cac thung va cost + heuristic1 cua chuoi hanh dong tuong ung
                actions.push(node_action + [action[-1]], cost(newActions[1:]) + heuristic2(newPosPlayer, newPosBox)) # Them vao actions chuoi hanh dong moi va cost + heuristic1 cua chuoi hanh dong tuong ung
    return temp # Tra ve mang luu cac hanh dong dan den goal

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'greedy':
        result = greedyFirstSearch(gameState)
    elif method == 'astar':
        result = aStarSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
