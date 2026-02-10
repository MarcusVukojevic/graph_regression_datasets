import os
import random
import javalang
import javalang.tree
import javalang.ast
import javalang.util
from javalang.ast import Node
from anytree import AnyNode, RenderTree
from anytree import find


edges={'Nexttoken':2,'Prevtoken':3,'Nextuse':4,'Prevuse':5,'If':6,'Ifelse':7,'While':8,'For':9,'Nextstmt':10,'Prevstmt':11,'Prevsib':12}
vocabdict = {'': 0, 'BasicType': 1, 'SuperMethodInvocation': 2, 'ForControl': 3, 'AnnotationMethod': 4, 'InferredFormalParameter': 5, 'LocalVariableDeclaration': 6, 'SuperConstructorInvocation': 7, 'Import': 8, 'ArraySelector': 9, 'BreakStatement': 10, 'FieldDeclaration': 11, 'EnumDeclaration': 12, 'ConstructorDeclaration': 13, 'Annotation': 14, 'ReferenceType': 15, 'EnhancedForControl': 16, 'TypeParameter': 17, 'Statement': 18, 'CompilationUnit': 19, 'EnumConstantDeclaration': 20, 'IfStatement': 21, 'ClassCreator': 22, 'SwitchStatement': 23, 'EnumBody': 24, 'PackageDeclaration': 25, 'Cast': 26, 'VariableDeclaration': 27, 'ArrayCreator': 28, 'This': 29, 'MethodReference': 30, 'InnerClassCreator': 31, 'InterfaceDeclaration': 32, 'FormalParameter': 33, 'CatchClauseParameter': 34, 'SynchronizedStatement': 35, 'VoidClassReference': 36, 'TypeArgument': 37, 'DoStatement': 38, 'Assignment': 39, 'ContinueStatement': 40, 'AssertStatement': 41, 'ExplicitConstructorInvocation': 42, 'AnnotationDeclaration': 43, 'StringLiteralExpr': 44, 'PrimitiveType': 45, 'TryStatement': 46, 'ElementArrayValue': 47, 'BlockStatement': 48, 'ClassReference': 49, 'ReturnStatement': 50, 'IntegerLiteralExpr': 51, 'TernaryExpression': 52, 'VariableDeclarator': 53, 'BinaryOperation': 54, 'ClassDeclaration': 55, 'TryResource': 56, 'MemberReference': 57, 'SuperMemberReference': 58, 'Literal': 59, 'CatchClause': 60, 'WhileStatement': 61, 'ElementValuePair': 62, 'ForStatement': 63, 'StatementExpression': 64, 'ConstantDeclaration': 65, 'ArrayInitializer': 66, 'MethodInvocation': 67, 'Modifier': 68, 'ThrowStatement': 69, 'LambdaExpression': 70, 'SwitchStatementCase': 71, 'MethodDeclaration': 72}
#from tokendic import vocabdict
import json
import re
"""
def get_token(node):
    token = ''
    #print(isinstance(node, Node))
    #print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
"""
def get_token(node):
    token = ''
    primitivetype = [ 'int', 'byte', 'short', 'long', 'float', 'double', 'boolean', 'char']
    #print(isinstance(node, Node))
    #print(type(node))
    if isinstance(node, str):
        if node in primitivetype:
            token = 'PrimitiveType'
        elif node.isnumeric():
            token = 'IntegerLiteralExpr'            
        else:
            token = 'StringLiteralExpr'
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children)) # list of tree objects 
    
def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)

def getnodes(node,nodelist):
    nodelist.append(node)
    children = get_child(node)
    for child in children:
        getnodes(child,nodelist)

class Queue():
    def __init__(self):
        self.__list = list()

    def isEmpty(self):
        return self.__list == []

    def push(self, data):
        self.__list.append(data)

    def pop(self):
        if self.isEmpty():
            return False
        return self.__list.pop(0)
    
def traverse(node,index):
    queue = Queue()
    queue.push(node)
    result = []
    while not queue.isEmpty():
        node = queue.pop()
        result.append(get_token(node))
        result.append(index)
        index+=1
        for (child_name, child) in node.children():
            #print(get_token(child),index)
            queue.push(child)
    return result

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)
def getnodeandedge_astonly(node,nodeindexlist,vocabdict,src,tgt):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        src.append(child.id)
        tgt.append(node.id)
        getnodeandedge_astonly(child,nodeindexlist,vocabdict,src,tgt)
def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)
def getedge_nextsib(node,vocabdict,src,tgt,edgetype):
    token=node.token
    for i in range(len(node.children)-1):
        src.append(node.children[i].id)
        tgt.append(node.children[i+1].id)
        edgetype.append([1])
        src.append(node.children[i+1].id)
        tgt.append(node.children[i].id)
        edgetype.append([edges['Prevsib']])
    for child in node.children:
        getedge_nextsib(child,vocabdict,src,tgt,edgetype)
def getedge_flow(node,vocabdict,src,tgt,edgetype,ifedge=False,whileedge=False,foredge=False):
    token=node.token
    if whileedge==True:
        if token=='WhileStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['While']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['While']])
    if foredge==True:
        if token=='ForStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['For']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['For']])
            '''if len(node.children[1].children)!=0:
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[0].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopstart'])
                src.append(node.children[1].children[-1].id)
                tgt.append(node.children[0].id)
                edgetype.append(edges['For_loopend'])
                src.append(node.children[0].id)
                tgt.append(node.children[1].children[-1].id)
                edgetype.append(edges['For_loopend'])'''
    #if token=='ForControl':
        #print(token,len(node.children))
    if ifedge==True:
        if token=='IfStatement':
            src.append(node.children[0].id)
            tgt.append(node.children[1].id)
            edgetype.append([edges['If']])
            src.append(node.children[1].id)
            tgt.append(node.children[0].id)
            edgetype.append([edges['If']])
            if len(node.children)==3:
                src.append(node.children[0].id)
                tgt.append(node.children[2].id)
                edgetype.append([edges['Ifelse']])
                src.append(node.children[2].id)
                tgt.append(node.children[0].id)
                edgetype.append([edges['Ifelse']])
    for child in node.children:
        getedge_flow(child,vocabdict,src,tgt,edgetype,ifedge,whileedge,foredge)
def getedge_nextstmt(node,vocabdict,src,tgt,edgetype):
    token=node.token
    if token=='BlockStatement':
        for i in range(len(node.children)-1):
            src.append(node.children[i].id)
            tgt.append(node.children[i+1].id)
            edgetype.append([edges['Nextstmt']])
            src.append(node.children[i+1].id)
            tgt.append(node.children[i].id)
            edgetype.append([edges['Prevstmt']])
    for child in node.children:
        getedge_nextstmt(child,vocabdict,src,tgt,edgetype)
def getedge_nexttoken(node,vocabdict,src,tgt,edgetype,tokenlist):
    def gettokenlist(node,vocabdict,edgetype,tokenlist):
        token=node.token
        if len(node.children)==0:
            tokenlist.append(node.id)
        for child in node.children:
            gettokenlist(child,vocabdict,edgetype,tokenlist)
    gettokenlist(node,vocabdict,edgetype,tokenlist)
    for i in range(len(tokenlist)-1):
            src.append(tokenlist[i])
            tgt.append(tokenlist[i+1])
            edgetype.append([edges['Nexttoken']])
            src.append(tokenlist[i+1])
            tgt.append(tokenlist[i])
            edgetype.append([edges['Prevtoken']])
def getedge_nextuse(node,vocabdict,src,tgt,edgetype,variabledict):
    def getvariables(node,vocabdict,edgetype,variabledict):
        token=node.token
        variable = ''
        variablenode = ''
        if token=='MemberReference':
            for child in node.children:
                if child.token==node.data.member:
                    variable=child.token
                    variablenode=child
            if variable != '' and variablenode  != '':
                
                if not variabledict.__contains__(variable):
                    variabledict[variable]=[variablenode.id]
                else:
                    variabledict[variable].append(variablenode.id)      
        for child in node.children:
            getvariables(child,vocabdict,edgetype,variabledict)
    getvariables(node,vocabdict,edgetype,variabledict)
    #print(variabledict)
    for v in variabledict.keys():
        for i in range(len(variabledict[v])-1):
                src.append(variabledict[v][i])
                tgt.append(variabledict[v][i+1])
                edgetype.append([edges['Nextuse']])
                src.append(variabledict[v][i+1])
                tgt.append(variabledict[v][i])
                edgetype.append([edges['Prevuse']])

def remove_comments(java_code):
    

    # Regular expression to remove comments
    #comment_regex = r"/\*[\s\S]*?\*/"
    #comment_regex = r'/\*{79}\n\s\*(?:.*\n)*?\s\*{79}/|/\*\n\s\*(?!\*)[^\n]*?\*/'
    comment_regex = r"/\*.*?\*/|/\*[a-zA-Z0-9]*?\*/"






    # Remove the comments from the Java code
    java_code_without_comments = re.sub(comment_regex, "", java_code,flags=re.S)

    return java_code_without_comments

def createast():
    asts=[]
    paths=[]
    alltokens=[]
    dirname = '/Users/samoaa/Downloads/Chalmers_PhD_files/Projects/Erorr_Analysis/Datasets/test_files/OSSBuilds/H2'
    count = 0
    sample = []
    for rt, dirs, files in os.walk(dirname):
        for file in files:
            try:
                programfile=open(os.path.join(rt,file),encoding='utf-8')
                programtext=programfile.read()
                #programtext=programtext.replace('\r','')
                #programtext=remove_comments(programtext)
                programtokens=javalang.tokenizer.tokenize(programtext)
                #print(list(programtokens))
                programast=javalang.parser.parse(programtokens)
                paths.append(os.path.join(rt,file))
                asts.append(programast)
                get_sequence(programast,alltokens)
                for token in alltokens:
                    if (token=='WhileStatement') and (file not in sample):
                        sample.append(file)
                programfile.close()
            except:
                print ("the file is not valid")
                print (file)
            #print(os.path.join(rt,file))
            
            #print(programast)
            #print(alltokens)
    astdict=dict(zip(paths,asts))
    ifcount=0
    whilecount=0
    forcount=0
    blockcount=0
    docount=0
    switchcount=0
    for token in alltokens:
        if token=='IfStatement':
            ifcount+=1
        if token=='WhileStatement':
            whilecount+=1
        if token=='ForStatement':
            forcount+=1
        if token=='BlockStatement':
            blockcount+=1
        if token=='DoStatement':
            docount+=1
        if token=='SwitchStatement':
            switchcount+=1
    print('IfStatement:',ifcount,'WhileStatement: ',whilecount,'ForStatement: ',forcount,'BlockStatement',blockcount,'DoStatement: ',docount,'SwitchStatement',switchcount)
    print('allnodes ',len(alltokens))
    alltokens=list(set(alltokens))      
    vocabsize = len(alltokens)
    #tokenids = range(vocabsize)
    #vocabdict = dict(zip(alltokens, tokenids))
    """for token in alltokens:
        if token not in vocabdict.keys():
            vocabdict[token] = len(vocabdict) + 1
    with open('tokendic.py', 'w') as fp:
        json.dump(vocabdict, fp)"""
    #tokenids = range(vocabsize)
    #vocabdict = dict(zip(alltokens, tokenids))
    print(vocabsize)
    return astdict,vocabsize,vocabdict

def createseparategraph(astdict,vocablen,vocabdict,device,mode='astandedges',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True):
    pathlist=[]
    treelist=[]
    print('nextsib ',nextsib)
    print('ifedge ',ifedge)
    print('whileedge ',whileedge)
    print('foredge ',foredge)
    print('blockedge ',blockedge)
    print('nexttoken', nexttoken)
    print('nextuse ',nextuse)
    print(len(astdict))
    i=0
    for path,tree in astdict.items():
        i = i+1
        #print(tree)
        #print(path)
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
      #  print (i,'--11')
        createtree(newtree, tree, nodelist)
       # print (i,'--22')
        #print(path)
        #print(newtree)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        if mode=='astonly':
           # print (i,'--333')
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
            #print (i,'--444')
        else:
           # print (i,'--555')
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)
           # print (i,'--666')
            if nextsib==True:
                #print (i,'--777')
                getedge_nextsib(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
                #print (i,'--888')
            getedge_flow(newtree,vocabdict,edgesrc,edgetgt,edge_attr,ifedge,whileedge,foredge)
            #print (i,'--999')
            if blockedge==True:
                #print (i,'--1010')
                getedge_nextstmt(newtree,vocabdict,edgesrc,edgetgt,edge_attr)
            tokenlist=[]
            if nexttoken==True:
                #print (i,'--1111')
                getedge_nexttoken(newtree,vocabdict,edgesrc,edgetgt,edge_attr,tokenlist)
                #print (i,'--1212')
            variabledict={}
            #print (i,'--1313')
            if nextuse==True:
                #print (i,'--1414')
                getedge_nextuse(newtree,vocabdict,edgesrc,edgetgt,edge_attr,variabledict)
                #print (i,'--1515')
        #x = torch.tensor(x, dtype=torch.long, device=device)
        #print (i,'--1616')
        edge_index=[edgesrc, edgetgt]
        #edge_index = torch.tensor([edgesrc, edgetgt], dtype=torch.long, device=device)
        astlength=len(x)
        #print(x)
        #print(edge_index)
        #print(edge_attr)
        pathlist.append(path)
        treelist.append([[x,edge_index,edge_attr],astlength])
        astdict[path]=[[x,edge_index,edge_attr],astlength]
    #treedict=dict(zip(pathlist,treelist))
    return astdict
if __name__ == '__main__':
    astdict, vocabsize, vocabdict=createast()
    AST = astdict.copy()
    fa_AST=createseparategraph(astdict, vocabsize, vocabdict,device='cpu',mode='else',nextsib=True,ifedge=True,whileedge=True,foredge=True,blockedge=True,nexttoken=True,nextuse=True)
'''

    tree =  {re.search(r"/([^/]+)$", key).group(1): value for key, value in AST.items()}
    import pandas as pd 
    df = pd.read_csv('/Users/samoaa/Downloads/Chalmers_PhD_files/Projects/Erorr_Analysis/Datasets/runtimes_data/OSSBuilds/run-time data/rdf4j/rdf4j_agg.csv', delimiter=';')
    df['Test case'] = df['Test case']+'.java'
    df = df[['Test case', 'Runtime in ms']]
    dictionary = df.set_index('Test case')['Runtime in ms'].to_dict()
    new_dict = {key: (tree[key], dictionary[key]) for key in tree if key in dictionary}
    test = {tree[key]: dictionary[key] for key in tree if key in dictionary}
'''
    
  



