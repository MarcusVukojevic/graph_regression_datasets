import argparse
import json
import os
import re

import javalang
from anytree import AnyNode
from javalang.ast import Node


# Edge type ids used to enrich plain AST edges with flow/sibling/token/use links.
edges = {
    'Nexttoken': 2,
    'Prevtoken': 3,
    'Nextuse': 4,
    'Prevuse': 5,
    'If': 6,
    'Ifelse': 7,
    'While': 8,
    'For': 9,
    'Nextstmt': 10,
    'Prevstmt': 11,
    'Prevsib': 12,
}

# Fixed vocabulary used by Type1/Type2 processing in this repository.
vocabdict = {
    '': 0, 'BasicType': 1, 'SuperMethodInvocation': 2, 'ForControl': 3,
    'AnnotationMethod': 4, 'InferredFormalParameter': 5, 'LocalVariableDeclaration': 6,
    'SuperConstructorInvocation': 7, 'Import': 8, 'ArraySelector': 9,
    'BreakStatement': 10, 'FieldDeclaration': 11, 'EnumDeclaration': 12,
    'ConstructorDeclaration': 13, 'Annotation': 14, 'ReferenceType': 15,
    'EnhancedForControl': 16, 'TypeParameter': 17, 'Statement': 18,
    'CompilationUnit': 19, 'EnumConstantDeclaration': 20, 'IfStatement': 21,
    'ClassCreator': 22, 'SwitchStatement': 23, 'EnumBody': 24,
    'PackageDeclaration': 25, 'Cast': 26, 'VariableDeclaration': 27,
    'ArrayCreator': 28, 'This': 29, 'MethodReference': 30, 'InnerClassCreator': 31,
    'InterfaceDeclaration': 32, 'FormalParameter': 33, 'CatchClauseParameter': 34,
    'SynchronizedStatement': 35, 'VoidClassReference': 36, 'TypeArgument': 37,
    'DoStatement': 38, 'Assignment': 39, 'ContinueStatement': 40,
    'AssertStatement': 41, 'ExplicitConstructorInvocation': 42,
    'AnnotationDeclaration': 43, 'StringLiteralExpr': 44, 'PrimitiveType': 45,
    'TryStatement': 46, 'ElementArrayValue': 47, 'BlockStatement': 48,
    'ClassReference': 49, 'ReturnStatement': 50, 'IntegerLiteralExpr': 51,
    'TernaryExpression': 52, 'VariableDeclarator': 53, 'BinaryOperation': 54,
    'ClassDeclaration': 55, 'TryResource': 56, 'MemberReference': 57,
    'SuperMemberReference': 58, 'Literal': 59, 'CatchClause': 60,
    'WhileStatement': 61, 'ElementValuePair': 62, 'ForStatement': 63,
    'StatementExpression': 64, 'ConstantDeclaration': 65, 'ArrayInitializer': 66,
    'MethodInvocation': 67, 'Modifier': 68, 'ThrowStatement': 69,
    'LambdaExpression': 70, 'SwitchStatementCase': 71, 'MethodDeclaration': 72
}

def get_token(node):
    """Map a javalang node (or literal) to the token name used by vocabdict."""
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
    """Return flattened children for a javalang AST node."""
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
    """Remove block comments to help parsing noisy source files."""
    comment_regex = r"/\*.*?\*/|/\*[a-zA-Z0-9]*?\*/"






    # Remove the comments from the Java code
    java_code_without_comments = re.sub(comment_regex, "", java_code,flags=re.S)

    return java_code_without_comments

def createast(dirname, strip_comments=False, only_java_files=True):
    """Parse a directory of Java files into a {path: javalang AST} dictionary."""
    if not os.path.isdir(dirname):
        raise FileNotFoundError(f"Input directory not found: {dirname}")

    asts = []
    paths = []
    all_tokens = []
    invalid_files = 0

    for rt, _dirs, files in os.walk(dirname):
        for file in files:
            if only_java_files and not file.endswith(".java"):
                continue

            file_path = os.path.join(rt, file)
            try:
                with open(file_path, encoding='utf-8') as programfile:
                    programtext = programfile.read()

                if strip_comments:
                    programtext = remove_comments(programtext)

                programtokens = javalang.tokenizer.tokenize(programtext)
                programast = javalang.parser.parse(programtokens)
                paths.append(file_path)
                asts.append(programast)
                get_sequence(programast, all_tokens)
            except Exception as exc:
                invalid_files += 1
                print(f"the file is not valid: {file_path} ({exc})")

    astdict = dict(zip(paths, asts))
    ifcount = sum(1 for token in all_tokens if token == 'IfStatement')
    whilecount = sum(1 for token in all_tokens if token == 'WhileStatement')
    forcount = sum(1 for token in all_tokens if token == 'ForStatement')
    blockcount = sum(1 for token in all_tokens if token == 'BlockStatement')
    docount = sum(1 for token in all_tokens if token == 'DoStatement')
    switchcount = sum(1 for token in all_tokens if token == 'SwitchStatement')

    print(
        'IfStatement:', ifcount,
        'WhileStatement:', whilecount,
        'ForStatement:', forcount,
        'BlockStatement', blockcount,
        'DoStatement:', docount,
        'SwitchStatement', switchcount
    )
    print('allnodes', len(all_tokens))
    print('valid_files', len(astdict), 'invalid_files', invalid_files)

    # This project uses a fixed token dictionary; vocabsize here is dataset coverage only.
    vocabsize = len(set(all_tokens))
    print('observed_vocab_size', vocabsize)
    return astdict, vocabsize, vocabdict

def createseparategraph(
    astdict,
    vocablen,
    vocabdict,
    device,
    mode='astandedges',
    nextsib=True,
    ifedge=True,
    whileedge=True,
    foredge=True,
    blockedge=True,
    nexttoken=True,
    nextuse=True,
):
    """Convert parsed AST objects to graph payloads expected by downstream code."""
    del vocablen  # preserved for backward compatibility with existing calls
    del device

    print('nextsib', nextsib)
    print('ifedge', ifedge)
    print('whileedge', whileedge)
    print('foredge', foredge)
    print('blockedge', blockedge)
    print('nexttoken', nexttoken)
    print('nextuse', nextuse)
    print('parsed_files', len(astdict))

    graph_dict = {}
    for path, tree in astdict.items():
        nodelist = []
        newtree = AnyNode(id=0, token=None, data=None)
        createtree(newtree, tree, nodelist)

        x = []
        edgesrc = []
        edgetgt = []
        edge_attr = []

        if mode == 'astonly':
            getnodeandedge_astonly(newtree, x, vocabdict, edgesrc, edgetgt)
        else:
            getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt, edge_attr)
            if nextsib:
                getedge_nextsib(newtree, vocabdict, edgesrc, edgetgt, edge_attr)
            getedge_flow(newtree, vocabdict, edgesrc, edgetgt, edge_attr, ifedge, whileedge, foredge)
            if blockedge:
                getedge_nextstmt(newtree, vocabdict, edgesrc, edgetgt, edge_attr)
            if nexttoken:
                tokenlist = []
                getedge_nexttoken(newtree, vocabdict, edgesrc, edgetgt, edge_attr, tokenlist)
            if nextuse:
                variabledict = {}
                getedge_nextuse(newtree, vocabdict, edgesrc, edgetgt, edge_attr, variabledict)

        edge_index = [edgesrc, edgetgt]
        astlength = len(x)
        graph_dict[path] = [[x, edge_index, edge_attr], astlength]

    return graph_dict


def _summarize_graphs(graph_dict):
    """Print a compact summary useful for smoke tests."""
    total_nodes = 0
    total_edges = 0
    for payload, ast_len in graph_dict.values():
        edge_index = payload[1]
        total_nodes += ast_len
        total_edges += len(edge_index[0])

    print('graphs_built', len(graph_dict))
    print('total_nodes', total_nodes)
    print('total_edges', total_edges)


def parse_args():
    parser = argparse.ArgumentParser(description='Build FA-AST graph representations from Java files.')
    parser.add_argument('--input-dir', required=True, help='Directory containing Java files to parse.')
    parser.add_argument('--output-json', default=None, help='Optional path where the generated graph dict is saved as JSON.')
    parser.add_argument('--mode', choices=['astandedges', 'astonly'], default='astandedges', help='Graph extraction mode.')
    parser.add_argument('--strip-comments', action='store_true', help='Strip block comments before tokenization.')
    parser.add_argument('--include-non-java', action='store_true', help='Also attempt to parse files without .java extension.')
    parser.add_argument('--disable-nextsib', action='store_true', help='Disable next-sibling edges.')
    parser.add_argument('--disable-ifedge', action='store_true', help='Disable if/ifelse edges.')
    parser.add_argument('--disable-whileedge', action='store_true', help='Disable while-loop edges.')
    parser.add_argument('--disable-foredge', action='store_true', help='Disable for-loop edges.')
    parser.add_argument('--disable-blockedge', action='store_true', help='Disable next-statement edges.')
    parser.add_argument('--disable-nexttoken', action='store_true', help='Disable next-token edges.')
    parser.add_argument('--disable-nextuse', action='store_true', help='Disable next-use edges.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    astdict, vocabsize, vocab = createast(
        dirname=args.input_dir,
        strip_comments=args.strip_comments,
        only_java_files=not args.include_non_java,
    )

    fa_ast = createseparategraph(
        astdict,
        vocabsize,
        vocab,
        device='cpu',
        mode=args.mode,
        nextsib=not args.disable_nextsib,
        ifedge=not args.disable_ifedge,
        whileedge=not args.disable_whileedge,
        foredge=not args.disable_foredge,
        blockedge=not args.disable_blockedge,
        nexttoken=not args.disable_nexttoken,
        nextuse=not args.disable_nextuse,
    )

    _summarize_graphs(fa_ast)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as fp:
            json.dump(fa_ast, fp)
        print('saved_json', args.output_json)
