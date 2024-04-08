import glob

import streamlit as st

import time
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import FewShotPromptTemplate

import openai
import gradio as gr

import random

from langchain.chains import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import ChatPromptTemplate

os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.org/v1"
os.environ["OPENAI_API_KEY"] = "sk-OKALqg46qdC65159Z5Wk4pWiGkbFwfiRWRVkvsudwUlikn6X"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4-1106-preview"
)

import langchain




def load_documents(directory="width.txt"):
    """
    加载books下的文件，进行拆分
    :param directory:
    :return:
    """
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # text_spliter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    text_spliter = CharacterTextSplitter(
        separator = "Q:",
        chunk_size = 0,
        chunk_overlap  = 0,
        is_separator_regex = True,
    )

    split_docs = text_spliter.split_documents(documents)
    # print(split_docs[:2])
    return split_docs

# load_documents


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
# load database
if not os.path.exists('VectorStore'):
    documents = load_documents()
    # documents_Rule = load_prompt_documents()
    db = store_chroma(documents, embeddings)

else:
    db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)


import  markdownify

def load_db1(question):




    width_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: NW width, when the space <= 0.235um, except INST region <= 0.365um
    The answer you should come up with:

    NW_W_4a 
    @ NW width, when the space <= 0.235um, except INST region <= 0.365um
    nw_meet_sps_sides = EXT [NW] <= 0.235 ABUT<90 OPPOSITE
    nw_meet_sps_block = NW WITH EDGE nw_meet_sps_sides
    err1 = (INT (nw_meet_sps_block) <= 0.365 ABUT<90 OPPOSITE) COIN EDGE nw_meet_sps_sides
    err2 = (nw_meet_sps_block WITH WIDTH > 0.365) WITH EDGE nw_meet_sps_sides
    err1 NOT INSIDE EDGE INST
    err2 NOT INSIDE INST


    Among them, NW_W_4a  represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets. Please note that the comment after @ must be inside curly brackets.
    In the SVRF language specification, the width check generally uses the keyword "INT". 
    "INT": Measures the separation between the interior sides of edges from the input layers. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default.
    "ABUT<90" which means that an error will be reported when the angle of two adjacent sides is less than 90 degrees, preventing layer1 and layer2 from directly touching each other without causing an error.
    "OPPOSITE": Specifies perpendicular extension of the measurement region from the edge, but not along the edge, is equal to your constraint. It converts the ¡°<¡± constraint to ¡°<=¡± when measuring intersecting edges. This causes output when intersecting edges abut at 90-degree angles. 
    "WITH EDGE": Selects polygons on the input layer that share an edge or edge segment on a second layer.
    "WITH WIDTH" : Constructs a layer containing polygons meeting the specified width constraint.
    "nw_meet_sps_block = NW WITH EDGE nw_meet_sps_sides" and "nw_meet_sps_sides = EXT [NW] <= 0.235 ABUT<90 OPPOSITE" are layer operations, used to construct the layer area required by the problem.
    " err1 NOT INSIDE EDGE INST" and "err2 NOT INSIDE INST" indicates that the error report should ignore the INST region, which corresponds to the "except INST region" in the question. 
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is: ALL_AA width in GATE poly direction for I/O region, except DMC7 region = 0.144+0.048*num
    What answer should you give:

    AA_W_3b 
    @ ALL_AA width in GATE poly direction for I/O region, except DMC7 region = 0.144+0.048*num
    err1 = (DFM COPY (DFM SPACE ALL_AA_IO < 0.144 BY INT VERTICAL) REGION) NOT INSIDE DMC7
    err1 NOT INSIDE SealR_NOT_BULK

    err2_EDGE = ALL_AA_convex_equal_two_v_edges TOUCH EDGE ALL_AA_IO
     err2_TMP = DFM PROPERTY err2_EDGE [-= ABS(REMAINDER((LENGTH(err2_EDGE) - 0.144), 0.048))] > 0
    err2 = ((DFM COPY err2_TMP) INSIDE EDGE DG) NOT INSIDE EDGE DMC7
     err2 NOT INSIDE EDGE SealR_NOT_BULK

    When my question is modified to: ALL_AA width in GATE poly direction for I/O region, except DMC7 region = 0.148+0.046*num
    Your expected answer is:

    AA_W_4b 
    @ ALL_AA width in GATE poly direction for I/O region, except DMC7 region = 0.148+0.046*num
    err1 = (DFM COPY (DFM SPACE ALL_AA_IO < 0.148 BY INT VERTICAL) REGION) NOT INSIDE DMC7
    err1 NOT INSIDE SealR_NOT_BULK

    err2_EDGE = ALL_AA_convex_equal_two_v_edges TOUCH EDGE ALL_AA_IO
    err2_TMP = DFM PROPERTY err2_EDGE [-= ABS(REMAINDER((LENGTH(err2_EDGE) - 0.148), 0.046))] > 0
    err2 = ((DFM COPY err2_TMP) INSIDE EDGE DG) NOT INSIDE EDGE DMC7
    err2 NOT INSIDE EDGE SealR_NOT_BULK


    When my question is modified to: ALL_AA width, except INST region >= 0.09um
    Your expected answer is: 
    AA_W_1 
    @ ALL_AA width, except INST region >= 0.09um
    err1 = INT ALL_AA < 0.09 ABUT<90 SINGULAR REGION
    err1 NOT INSIDE INST


    When my question is modified to: ALL_GT width in core region, except GT_P96, INST, MARKS, OCOVL and LOGO regions = 0.016/0.018/0.02, 0.032, 0.07~0.242um
    Your expected answer is:

    GT_W_3 
    @ ALL_GT width in core region, except GT_P96, INST, MARKS, OCOVL and LOGO regions = 0.016/0.018/0.02, 0.032, 0.07~0.242um
    except_region = OR GT_P96 OCOVL LOGO INST MARKS
    chk_GT = (ALL_GT NOT INSIDE DG) NOT INSIDE except_region
    good_1 = OR ALL_GT_016 ALL_GT_018 ALL_GT_020
    good_2 = WITH WIDTH chk_GT == 0.032
    good_3 = WITH WIDTH chk_GT >= 0.070 <= 0.242
    err1 = (chk_GT NOT (OR good_1 good_2 good_3)) NOT (OR DG except_region)
    err1 NOT INSIDE SealR_NOT_BULK



    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    space_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. Later, I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: Space between ACTIVE AA and pick-up AA in S/D direction, except DSTR and INST regions >= 0.18um
    The answer you should come up with:\n
    AA_S_10 
    @ Space between ACTIVE AA and pick-up AA in S/D direction, except DSTR and INST regions >= 0.18um
    chk_pick_up = ((TAP_RAW NOT DSTR) NOT COIN EDGE ACT) COIN EDGE AA_v_edges
    err1 = EXT ACT chk_pick_up < 0.180 ABUT<90 OPPOSITE REGION
    err1 NOT INSIDE (OR DSTR INST)\n
    Among them,AA_S_10  represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.Please note that the comment after @ must be inside curly brackets.
    "chk_pick_up = ((TAP_RAW NOT DSTR) NOT COIN EDGE ACT) COIN EDGE AA_v_edges" indicates layer operations. 
    "chk_pick_up" corresponds to the "pick-up AA in S/D direction" layer in the question. "ACT" means "ACTIVE AA", which corresponds to the "ACTIVE AA" in the question. 
    In the SVRF language specification, the space check generally uses the keyword "EXT". 
    "EXT": Measures the separation between the exterior sides of edges from the input layers. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default. This operation is polygon-directed if you use the REGION keyword, edgedirected if you use the [ ] and ( ) operators, and error-directed by default.\n
    "err1 NOT INSIDE (OR DSTR INST)" indicates that the error report should ignore the DSTR and INST region, which corresponds to the "except DSTR and INST regions" in the question. 
    "ABUT<90" which means that an error will be reported when the angle of two adjacent sides is less than 90 degrees, preventing layer1 and layer2 from directly touching each other without causing an error.
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is:Space between Mxy when one or both width >= 0.542um, except LOGO region >= 0.188um
    What answer should you give:\n
    M3_S_7 
    @ Space between Mxy when one or both width >= 0.542um, except LOGO region >= 0.188um
    chk_Mn = M3 WITH WIDTH >= 0.542
    err1 = EXT M3 chk_Mn < 0.188 ABUT<90 REGION
    err1 NOT INSIDE LOGO\n
    When my question is modified to:Space between Mxy when one or both width >= 0.52um, except INST region >= 0.148um
    Your expected answer is:\n
    M3_S_7 
    @ Space between Mxy when one or both width >= 0.52um, except INST region >= 0.188um
    chk_Mn = M3 WITH WIDTH >= 0.52
    err1 = EXT M3 chk_Mn < 0.148 ABUT<90 REGION
    err1 NOT INSIDE INST\n
    When my question is modified to:Space between ALL_AA, except INST region >= 0.09um
    Your expected answer is: \n
    AA_S_1 
    @ Space between ALL_AA, except INST region >= 0.09um
    err1 = EXT ALL_AA < 0.09 ABUT<90 SINGULAR REGION
    err1 NOT INSIDE INST\n
    When my question is modified to:Core_NW space, when the width <= 0.235um, expect INST region <= 0.365um
    Your expected answer is:

    NW_S_2_e 
    @ Core_NW space, when the width <= 0.235um, except INST region <= 0.365um
    nw_meet_wid_sides = INT [core_NW] <= 0.235 ABUT<90 OPPOSITE
    nw_meet_sps_sides = EXT [core_NW] <= 0.365 ABUT<90 OPPOSITE
    nw_errs_sps_sides = EXT (core_NW) <= 0.365 ABUT<90 OPPOSITE
    err1 = INT (nw_meet_wid_sides NOT COIN EDGE nw_meet_sps_sides) nw_errs_sps_sides <= 0.235 OPPOSITE REGIONS
    err1 NOT INSIDE INST
  
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    Area_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    You need to convert the entered rule requirements into corresponding SVRF codes.\n
    In the SVRF language specification, the area check generally uses the keyword "AREA". \n
    For example, err1 = AREA AOP_AA < 0.0138 means the area of AOP_AA >= 0.0138um2. \n
    Below is an example of a rule requirement and the corresponding SVRF code forming a question and answer pair:
    Q: AOP_AA area in I/O region, except DMC7 region >= 0.0285um2
    A:
    AA_A_2 
    @ AOP_AA area in I/O region, except DMC7 region >= 0.0285um2
    err1 = AREA (AOP_AA NOT OUTSIDE DG) < 0.0285
    err1 NOT INSIDE DMC7
 
    "Q:" is followed by rule requirements.\n
    "A:" is followed by the corresponding SVRF code. \n
    In this example the rule  name is called AA_A_2. \n
    The content after the rule name must be enclosed in curly brackets. \n
    The content after @ is a comment, which is the same as the rule requirements. Within a rule requirement, comments should be written first. \n
    "err1 = AREA (AOP_AA NOT OUTSIDE DG) < 0.0285" means that if the AOP_AA area in I/O region is less than 0.0285um2, an error will be reported.\n
    "err1 NOT INSIDE DMC7" indicates that the error report should ignore the DMC7 region, which corresponds to the "except DMC7 region" in the rule requirements. \n
    This forms an SVRF code corresponding to the rule file.When you answer, you only need to return the corresponding code code and do not need to answer anything else.\n
    {input}
    woshishei
    """

    Density_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. Later, I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: Space between ACTIVE AA and pick-up AA in S/D direction, except DSTR and INST regions >= 0.18um
    The answer you should come up with:
    AA_S_10 
    @ Space between ACTIVE AA and pick-up AA in S/D direction, except DSTR and INST regions >= 0.18um
    chk_pick_up = ((TAP_RAW NOT DSTR) NOT COIN EDGE ACT) COIN EDGE AA_v_edges
    err1 = EXT ACT chk_pick_up < 0.180 ABUT<90 OPPOSITE REGION
    err1 NOT INSIDE (OR DSTR INST)
    Among them,AA_S_10  represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.
    "chk_pick_up = ((TAP_RAW NOT DSTR) NOT COIN EDGE ACT) COIN EDGE AA_v_edges" indicates layer operations. 
    "chk_pick_up" corresponds to the "pick-up AA in S/D direction" layer in the question. "ACT" means "ACTIVE AA", which corresponds to the "ACTIVE AA" in the question. 
    In the SVRF language specification, the space check generally uses the keyword "EXT". 
    "EXT": Measures the separation between the exterior sides of edges from the input layers. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default. This operation is polygon-directed if you use the REGION keyword, edgedirected if you use the [ ] and ( ) operators, and error-directed by default.
    "err1 NOT INSIDE (OR DSTR INST)" indicates that the error report should ignore the DSTR and INST region, which corresponds to the "except DSTR and INST regions" in the question. 
    "ABUT<90" which means that an error will be reported when the angle of two adjacent sides is less than 90 degrees, preventing layer1 and layer2 from directly touching each other without causing an error.
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is:Space between Mxy when one or both width >= 0.542um, except LOGO region >= 0.188um
    What answer should you give:
    M3_S_7 
    @ Space between Mxy when one or both width >= 0.542um, except LOGO region >= 0.188um
     chk_Mn = M3 WITH WIDTH >= 0.542
     err1 = EXT M3 chk_Mn < 0.188 ABUT<90 REGION
     err1 NOT INSIDE LOGO
    When my question is modified to:Space between Mxy when one or both width >= 0.52um, except INST region >= 0.148um
    Your expected answer is:
    M3_S_7 
    @ Space between Mxy when one or both width >= 0.52um, except INST region >= 0.188um
     chk_Mn = M3 WITH WIDTH >= 0.52
     err1 = EXT M3 chk_Mn < 0.148 ABUT<90 REGION
     err1 NOT INSIDE INST
    When my question is modified to:Space between ALL_AA, except INST region >= 0.09um
    Your expected answer is: 
    AA_S_1 
    @ Space between ALL_AA, except INST region >= 0.09um
     err1 = EXT ALL_AA < 0.09 ABUT<90 SINGULAR REGION
     err1 NOT INSIDE INST
    When my question is modified to:Core_NW space, when the width <= 0.235um, expect INST region <= 0.365um
    Your expected answer is:
    
    NW_S_2_e 
    @ Core_NW space, when the width <= 0.235um, expect INST region <= 0.365um
     nw_meet_wid_sides = INT [core_NW] <= 0.235 ABUT<90 OPPOSITE
     nw_meet_sps_sides = EXT [core_NW] <= 0.365 ABUT<90 OPPOSITE
     nw_errs_sps_sides = EXT (core_NW) <= 0.365 ABUT<90 OPPOSITE
     err1 = INT (nw_meet_wid_sides NOT COIN EDGE nw_meet_sps_sides) nw_errs_sps_sides <= 0.235 OPPOSITE REGION
     err1 NOT INSIDE INST
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    Extension_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: HVT_N extension outside of ALL_AA in GATE poly direction >= 0.048um
    The answer you should come up with:
    
    HVT_N_EX_2 
    @ HVT_N extension outside of ALL_AA in GATE poly direction >= 0.048um
     err1 = ENC ALL_AA_h_edges HVT_N < 0.048 ABUT<90 OPPOSITE REGION
     err1 NOT INSIDE SealR_NOT_BULK
    
    Among them, HVT_N_EX_2 represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.
    In the SVRF language specification, the extension check generally uses the keyword "ENC". 
    "ENC": Measures the separation between the exterior sides of edges from one layer and the interior sides of edges from another layer. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default. This operation is polygondirected if you use the REGION keyword, edge-directed if you use the [ ] and ( ) operators, and error-directed by default. 
    "HVT_N" and "ALL_AA_h_edges" correspond to "HVT_N" and "ALL_AA in GATE poly direction" in the question respectively. 
    "err1 NOT INSIDE SealR_NOT_BULK" indicates that the error report should ignore the SealR_NOT_BULK region. 
    "ABUT<90" which means that an error will be reported when the angle of two adjacent sides is less than 90 degrees, preventing layer1 and layer2 from directly touching each other without causing an error.
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is: HVT_N extension outside of ALL_AA (vertical edge abut AA edge is allowed) >= 0.045um
    What answer should you give:
    
    HVT_N_EX_1 
    @ HVT_N extension outside of ALL_AA (vertical edge abut AA edge is allowed) >= 0.045um
     CHECK_RAW = ENC ALL_AA HVT_N < 0.045 ABUT<90 SINGULAR
     WAIVE_CRN = HVT_N COIN INSIDE EDGE ALL_AA_v_edges
     err1 = DFM COPY (DFM PROPERTY CHECK_RAW WAIVE_CRN ABUT ALSO OVERLAP MULTI SPLIT [-= count(WAIVE_CRN)] == 0) REGION
     err1 NOT INSIDE SealR_NOT_BULK
    
    When my question is modified to:SVT_P extension outside of (ALL_GT NOT P2) (ALL_GT width > 0.09um) >= 0.058um
    Your expected answer is:
    
    SVT_P_EX_5 
    @ SVT_P extension outside of (ALL_GT NOT P2) (ALL_GT width > 0.09um) >= 0.058um
     err1 = ENC ALL_GT_NOT_P2_gt_090 SVT_P < 0.058 ABUT<90 SINGULAR REGION
     err1 NOT INSIDE SealR_NOT_BULK
    
    When my question is modified to:AR extension outside of AA in GATE poly direction (extension <= 0 is not allowed) = 0.048+0.048*num
    Your expected answer is: 
    AR_EX_1 
    @ AR extension outside of AA in GATE poly direction (extension <= 0 is not allowed) = 0.048+0.048*num
     chk_AR = AR_v_edges_outer_AA OUTSIDE EDGE AA  //;AR whole edge on STI besides the edge GATE will also be checked (not need to exclude it based on FIN pitch)
     err1 = DFM PROPERTY chk_AR [-= ABS(REMAINDER((LENGTH(chk_AR) - 0.048),0.048))] > 0
     err1 NOT INSIDE EDGE SealR_NOT_BULK
    
     err2 = ENC AA AR_h_edges < 0.048 ABUT<90 OPPOSITE REGION INSIDE ALSO
     err2 NOT INSIDE SealR_NOT_BULK
    
    When my question is modified to:M0G (width = 0.06um) extension of DIR in DIR width direction at both sides (extension <= 0um is not allowed) = 0.014um
    Your expected answer is:
    
    HR_EX_1 
    @ M0G (width = 0.06um) extension of DIR in DIR width direction at both sides (extension <= 0um is not allowed) = 0.014um
     err1 = INT (HR_M0G_60 NOT OUTSIDE EDGE DIR) < 0.002 ABUT==90 INTERSECTING ONLY REGION
     err1 NOT INSIDE SealR_NOT_BULK
    
     good_enc = ENC [DIR_L_edges] HR_M0G_60 == 0.014 ABUT<90 OPPOSITE
     err2 = (DIR_L_edges INSIDE EDGE HR_M0G_60) NOT COIN EDGE good_enc
     err2 NOT INSIDE EDGE SealR_NOT_BULK
    
    
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    Enclosure_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: CELLB enclosure of ALL_AA in S/D direction, except DMCB1 region >= 0.08um
    The answer you should come up with:
    CELLB_EN_1 
    @ CELLB enclosure of ALL_AA in S/D direction, except DMCB1 region >= 0.08um
     err1 = ENC (ALL_AA AND CELLB) CELLB_v_edges < 0.08 ABUT<90 OPPOSITE REGION
     err1 NOT INSIDE DMCB1
    
    Among them, CELLB_EN_1 represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.
    In the SVRF language specification, the enclosure check generally uses the keyword "ENC". 
    "ENC": Measures the separation between the exterior sides of edges from one layer and the interior sides of edges from another layer. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default. This operation is polygondirected if you use the REGION keyword, edge-directed if you use the [ ] and ( ) operators, and error-directed by default. 
    "CELLB_v_edges" and "(ALL_AA AND CELL B)" correspond to "CELLB" and "ALL_AA in S/D direction" in the question respectively. 
    "err1 NOT INSIDE DMCB1" indicates that the error report should ignore the DMCB1 region, which corresponds to the "except DMCB1 region" in the question. 
    "ABUT<90" which means that an error will be reported when the angle of two adjacent sides is less than 90 degrees, preventing layer1 and layer2 from directly touching each other without causing an error.
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is:CELLB enclosure of ALL_AA in GATE poly direction, except DMCB1 region = 0.048+0.048*num 
    What answer should you give:
    CELLB_EN_2 
    @ CELLB enclosure of ALL_AA in GATE poly direction, except DMCB1 region = 0.048+0.048*num
     err1 = ENC ALL_AA_h_edges CELLB < 0.048 ABUT<90 OPPOSITE REGION  //;ouside AA covered by CELLB.R.1
     err1 NOT INSIDE DMCB1
    
     err2 = DFM COPY (OFFGRID (ALL_AA_h_edges NOT INSIDE EDGE DMCB1) 1 480 OFFSET 0 480 INSIDE OF LAYER CELLB ABSOLUTE) REGION
     err2 NOT INSIDE DMCB1
    
    When my question is modified to:CELLB enclosure of ALL_AA in GATE poly direction, except INST region = 0.048+0.048*num 
    Your expected answer is:
    CELLB_EN_2 
    @ CELLB enclosure of ALL_AA in GATE poly direction, except INST region = 0.048+0.048*num
     err1 = ENC ALL_AA_h_edges CELLB < 0.048 ABUT<90 OPPOSITE REGION  //;ouside AA covered by CELLB.R.1
     err1 NOT INSIDE DMCB1
    
     err2 = DFM COPY (OFFGRID (ALL_AA_h_edges NOT INSIDE EDGE DMCB1) 1 480 OFFSET 0 480 INSIDE OF LAYER CELLB ABSOLUTE) REGION
     err2 NOT INSIDE INST
    
    When my question is modified to:The outmost FIN enclosure by ALL_AA in GATE poly direction, except LDBK, DMCMK1, INST and MARKS regions = 0.019um
    Your expected answer is: 
    AA_EN_2 
    @ The outmost FIN enclosure by ALL_AA in GATE poly direction, except LDBK, DMCMK1, INST and MARKS regions = 0.019um
     WAIVE_AREA = OR INST_LDBK DMCMK1 MARKS
     chk_EDGE = ANGLE ((EXPAND EDGE ALL_AA_h_edges INSIDE BY 0.019) NOT COIN EDGE ALL_AA) == 0
     err1 = chk_EDGE NOT COIN EDGE FIN
     err1 NOT INSIDE EDGE WAIVE_AREA
    
     err2 = ALL_AA_h_edges NOT OUTSIDE EDGE FIN
     err2 NOT INSIDE EDGE WAIVE_AREA
    
    When my question is modified to:P+AA enclosure by NW, except INST and DIOMK2 regions >= 0.048um
    Your expected answer is:
    
    AA_EN_1a 
    @ P+AA enclosure by NW, except INST and DIOMK2 regions >= 0.048um
     err1 = ENC (PAA AND NW) NW < 0.048 ABUT<90 REGION
     err1 NOT INSIDE INST_DIOMK2
    
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    Overlap_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: AA overlap AR in S/D direction = 0.01, 0.02um
    The answer you should come up with:
    
    AR_O_1 
    @ AA overlap AR in S/D direction = 0.01, 0.02um
     err1 = DFM COPY (DFM SPACE AA AR < 0.01 BY INT HORIZONTAL) REGION
     err2 = DFM COPY (DFM SPACE AA AR > 0.01 < 0.02 BY INT HORIZONTAL) REGION
     err1 NOT INSIDE SealR_NOT_BULK
     err2 NOT INSIDE SealR_NOT_BULK
    
    
    Among them, AR_O_1 represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.
    keyword "INT": Measures the separation between the interior sides of edges from the input layers. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default. 
    This operation is polygon-directed if you use the REGION keyword, edgedirected if you use the [ ] and ( ) operators, and error-directed by default.
    keyword "DFM SPACE": Allows for very efficient measurements of edge-to-edge distances for large (>0.5 um) distance constraints. 
    This operation performs measurements and produces results that are similar to the DRC operations External, Internal, and Enclosure. 
    The DFM Space operation supports a much more limited set of options than these operations. The output layer is always a derived error layer (the edge output of the External, Internal and Enclosure operations is not supported).
    keyword "DFM COPY":Copies layers of any type to a new layer, and allows selection of specific cells to copy.
    " NOT INSIDE SealR_NOT_BULKL" indicates that the error report should ignore the SealR_NOT_BULKL region. 
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is: Overlap of NW and DNW >= 0.358um
    What answer should you give:
    
    DNW_O_1 
    @ Overlap of NW and DNW >= 0.358um
     err1 = INT NW DNW_in_NW_hole < 0.358 ABUT<90 SINGULAR REGION
     err1 NOT INSIDE SealR_NOT_BULK
    
     err2 = DNW_in_NW_hole NOT INSIDE EDGE NW
     err2 NOT INSIDE EDGE SealR_NOT_BULK
    
     err3 = INT NW (DNW NOT DNW_in_NW_hole) < 0.358 ABUT<90 SINGULAR REGION
     err3 NOT INSIDE SealR_NOT_BULK
    
    
    When my question is modified to:DG overlap of NW (abut is allowed) >= 0.238um
    Your expected answer is:
    
    DG_O_1 
    @ DG overlap of NW (abut is allowed) >= 0.238um
     err1 = INT DG NW < 0.238 ABUT>0<90 SINGULAR REGION
     err1 NOT INSIDE SealR_NOT_BULK
    
    
    When my question is modified to: Overlap of (ALL_GT NOT P2) and LFN_P, except small ALL_GT jogs <= 0.004um, Dummy_Cell_WO_IMP, and LFN_P vertical edge CUT (GTMK1 OR AR) (channel length <= 0.024um, centerline abut LFN_P vertical edge) >= 0.083um
    Your expected answer is: 
    LFN_P_O_1 
    @ Overlap of (ALL_GT NOT P2) and LFN_P, except small ALL_GT jogs <= 0.004um, Dummy_Cell_WO_IMP, and LFN_P vertical edge CUT (GTMK1 OR AR) (channel length <= 0.024um, centerline abut LFN_P vertical edge) >= 0.083um
     y1 = LFN_P_in_all_GT_in_ar_gtmk1_edges COIN EDGE all_GT_center_even
     y2 = ANGLE (LFN_P_in_all_GT_in_ar_gtmk1_edges TOUCH EDGE (LFN_P_in_all_GT_in_ar_gtmk1_area WITH EDGE (LFN_P_in_all_GT_in_ar_gtmk1_area COIN EDGE y1))) == 0
     y3 = LFN_P_in_ALL_GT_h_edges NOT COIN EDGE y2
     err1 = INT y3 ALL_GT_NOT_P2_not_jog_lteq_004 < 0.083 ABUT<90 OPPOSITE REGION
     err1 NOT INSIDE Dummy_Cell_WO_IMP
    
    
    When my question is modified to: M0_B1 overlap M0 in GATE poly direction and S/D direction respectively = 0.022/0.040um
    Your expected answer is:
    
    BM0_O_1 
    @ M0_B1 overlap M0 in GATE poly direction and S/D direction respectively = 0.022/0.040um
     err1 = INT M0_B1_h_edges M0i < 0.022 ABUT>0<90 OPPOSITE REGION //;exclude butted M0_B1 and M0
     err2 = INT M0_B1_v_edges M0i < 0.040 ABUT>0<90 OPPOSITE REGION
     err1 NOT INSIDE SealR_NOT_BULK
     err2 NOT INSIDE SealR_NOT_BULK
    
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    Length_template = """You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. I will give you a question, and then you need to return the corresponding code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: AR length in GATE poly direction, except OCOVL region >= 0.24um
    The answer you should come up with:
    
    AR_L_1 
    @ AR length in GATE poly direction, except OCOVL region >= 0.24um
     err1 = LENGTH AR_v_edges < 0.240
     err1 NOT INSIDE EDGE OCOVL
    
    Among them, AR_L_1 represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.
    In the SVRF language specification, the length check generally uses the keyword "LENGTH". 
    "LENGTH": Selects edges with a length that satisfies the given constraint. 
    " err1 NOT INSIDE EDGE OCOVL" indicates that the error report should ignore the OCOVL region, which corresponds to the "except OCOVL region" in the question. 
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is: AOP_GT jog length (jog width <= 0.004um) >= 0.24um
    What answer should you give:
    
    GT_L_4 
    @ AOP_GT jog length (jog width <= 0.004um) >= 0.24um
     edge_waive = CONVEX EDGE AOP_GT ANGLE1 == 90 LENGTH1 >= 0.24 ANGLE2 == 270 LENGTH2 >= 0.24 WITH LENGTH <= 0.004
     edge_check = LENGTH AOP_GT_convex_90_270_edge <= 0.004
     err1 = edge_check NOT COIN EDGE edge_waive
     err1 NOT INSIDE EDGE SealR_NOT_BULK
    
    When my question is modified to:P2 edge length in S/D direction >= 0.09um
    Your expected answer is:
    
    P2_L_1 
    @ P2 edge length in S/D direction >= 0.09um
     err1 = LENGTH P2_h_edges < 0.09
     err1 NOT INSIDE EDGE SealR_NOT_BULK
    
    When my question is modified to:M0 length (width = 0.04um), except (M0 interact TRCMK) and OCOVL region >= 0.114um
    Your expected answer is: 
    M0_L_1 
    @ M0 length (width = 0.04um), except (M0 interact TRCMK) and OCOVL region >= 0.114um
     CHK_M0 = M0 NOT INTERACT TRCMK
     err1 = LENGTH (INT [CHK_M0] == 0.04 OPPOSITE) < 0.04
     err2 = LENGTH (M0_040 INTERACT CHK_M0) > 0.04 < 0.114
     err3 = RECTANGLE CHK_M0 ASPECT == 1
     err1 NOT INSIDE EDGE OCOVL
     err2 NOT INSIDE EDGE OCOVL
     err3 NOT INSIDE OCOVL
    
    
    When my question is modified to:AOP_M0C edge length in S/D direction >= 0.09um
    Your expected answer is:
    
    M0C_L_1 
    @ AOP_M0C edge length in S/D direction >= 0.09um
     err1 = LENGTH AOP_M0C_h_edges < 0.090
     err1 NOT INSIDE EDGE SealR_NOT_BULK
    
    
    
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    rule_template="""You are a word processing expert. In the following text, there is a string: woshishei. 
    Find this string, remove it, and output the rest. Please remember that your goal is to find and remove a string, not programming, nor exploring the SVRF language. 
    If all you have to do is remove one string and output the rest, I will give you $200. Here is the text you need to process:
    I need your help to write a programming language for a specific field. I will give you a question, and then you need to return the corresponding code code. 
    suppose you are an expert in the SVRF(Standard Verification Rule Format) language. 
    You need to convert the entered rule requirements into corresponding SVRF codes.
    Here is an example: My question: FIN must be an orthogonal rectangle
    The answer you should come up with:
    
    FIN_R_1 
    @ FIN must be an orthogonal rectangle
     err1 = NOT RECTANGLE FIN ORTHOGONAL ONLY
     err1 NOT INSIDE SealR_NOT_BULK
    
     err2 = INT FIN < 0.001 ABUT<90 SINGULAR REGION
     err2 NOT INSIDE SealR_NOT_BULK
    
    
    Among them, FIN_R_1 represents the rule name. @ is followed by a comment which is the same as the question asked. 
    The content after the rule name must be enclosed in curly brackets.
    keyword "INT": Measures the separation between the interior sides of edges from the input layers. Measured edge pairs that satisfy the given constraint are output. Intersecting edge pairs are not measured by default. 
    This operation is polygon-directed if you use the REGION keyword, edgedirected if you use the [ ] and ( ) operators, and error-directed by default.
    " NOT INSIDE SealR_NOT_BULKL" indicates that the error report should ignore the SealR_NOT_BULKL region. 
    I will make numerical modifications to the question being asked. Here is an example: 
    When my question is: NPAA in chip design is not allowed
    What answer should you give:
    
    AA_R_2 
    @ NPAA in chip design is not allowed
     err1 = NPAA INTERACT DRC:1
     err1 NOT INSIDE SealR_NOT_BULK
    
    When my question is modified to:(ALL_AA interact GT (width >= 0.032um)) CUT SVT_N is not allowed
    Your expected answer is:
    
    SVT_N_R_1 
    @ (ALL_AA interact GT (width >= 0.032um)) CUT SVT_N is not allowed
     err1 = ALL_AA_INTERACT_GT_eqgt_032 CUT SVT_N
     err1 NOT INSIDE SealR_NOT_BULK
    
    
    When my question is modified to: M0 (width = 0.042um) must inside GT_P96
    Your expected answer is: 
    M0_R_9 
    @ M0 (width = 0.042um) must inside GT_P96
     err1 = M0_042 NOT GT_P96
     err1 NOT INSIDE SealR_NOT_BULK
    
    
    When my question is modified to: It is not allowed M0 (M0 width = 0.04/0.042um) ¦¤L > 0.1um, except single M0 pickup case
                                                        DRC highlight M0 segment (length > 0.1um) without an adjacent M0
    Your expected answer is:
    M0_R_11_DFM1 
    @ It is not allowed M0 (M0 width = 0.04/0.042um) ¦¤L > 0.1um, except single M0 pickup case
    @ DRC highlight M0 segment (length > 0.1um) without an adjacent M0
     iso_m0_a = INT M0_040_042_L_edges_eq_1_pitch M0_040_042_L_edges_lt_3_pitch >= 0.040 <= 0.042 ABUT<90 OPPOSITE REGION
     iso_m0_b = INT M0_040_042_L_edges_gt_1_pitch >= 0.040 <= 0.042 ABUT<90 OPPOSITE REGION
     err1 = (iso_m0_a ENCLOSE RECTANGLE 0.002 0.100+GLOBAL_TOLERANCE ORTHOGONAL ONLY) NOT M0_R_11_waive_one_M0_pkup
     err2 = (iso_m0_b ENCLOSE RECTANGLE 0.002 0.100+GLOBAL_TOLERANCE ORTHOGONAL ONLY) NOT M0_R_11_waive_one_M0_pkup
     err1 NOT INSIDE SealR_NOT_BULK
     err2 NOT INSIDE SealR_NOT_BULK
    
    When you answer, you only need to return the corresponding code code and do not need to answer anything else.The above are just some examples that I have taught you how to modify. I want you to address the issue of outputting code below.
    The above example is just one scenario, telling you how to modify these rule names and specific numbers in the rules. The question I am asking below may not belong to this situation. You should refer to the examples extracted from the local knowledge base below to make modifications.This is an example extracted from the local knowledge base, similar to my question. You should refer to the practical issues and modify the values.
    Finally, I would like to ask you to pay attention to the formatting issue. Due to some reasons, these codes need to be enclosed in parentheses similar to C language, but the example I gave you was not written. When giving the answer, please use front parentheses after the rule name and back parentheses at the end of the code to enclose the code.
    You need to provide the corresponding code for the question
    {input}
    woshishei
    """

    prompt_info = [
        {
            "name": "width rule",
            "description": "Choose this rule when the main keyword of the sentence is width.",
            "prompt_template": width_template
        },
        {
            "name": "space rule",
            "description": "Choose this rule when the main keyword of the sentence is space.",
            "prompt_template": space_template
        },
        {
            "name": "Area rule",
            "description": "Choose this rule the main keyword of the sentence is area.",
            "prompt_template": Area_template
        },
        {
            "name": "Density rule",
            "description": "Choose this rule when the main keyword of the sentence is density.",
            "prompt_template": Density_template
        },
        {
            "name": "Extension rule",
            "description": "Choose this rule when the main keyword of the sentence is extension.",
            "prompt_template": Extension_template
        },
        {
            "name": "Enclosure rule",
            "description": "Choose this rule when the main keyword of the sentence is enclosure.",
            "prompt_template": Enclosure_template
        },
        {
            "name": "Overlap rule",
            "description": "Choose this rule when the main keyword of the sentence is overlap.",
            "prompt_template": Overlap_template
        },
        {
            "name": "Length rule",
            "description": " Choose this rule when the main keyword of the sentence is length.",
            "prompt_template": Length_template
        },
        {"name": "rule's rule",
            "description": "Choose this rule when there are no eight words in the sentence: space, width, length, enclosure, area, extension, overlap, and density",
            "prompt_template": rule_template

        },
    ]

    destination_chains = {}

    for p_info in prompt_info:

       name = p_info["name"]
       prompt_template = p_info["prompt_template"]
       prompt = ChatPromptTemplate.from_template(template=prompt_template)

       chain = LLMChain(llm=llm, prompt=prompt)

       destination_chains[name] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_info]

    destinations_str = "\n".join(destinations)

    MULTI_PROMPT_ROUTER_TEMPLATE = """ Given a raw text input to a \  
    language model select the model prompt best suited for the input. \  
    You will be given the names of the available prompts and a \  
    description of what the prompt is best suited for. \  

    << FORMATTING >>  
    Return a markdown code snippet with a JSON object formatted to look like:  
     ```json  {{{{    "destination": string \ name of the prompt to use or "DEFAULT"    "next_inputs": string \ Completely unmodified  version of the original input  }}}}  ```  
    REMEMBER: "destination" MUST be one of the candidate prompt \  
    names specified below OR it can be "DEFAULT" if the input is not\  
    well suited for any of the candidate prompts.  
    REMEMBER: "next_inputs" must be the original input \  
 
    << CANDIDATE PROMPTS >>  
    {destinations}  
    << INPUT >>  
    {{input}}  
    << OUTPUT (remember to include the ```json)>> 
    """

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)

    router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser(),)

    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True
        )

    last_result = chain.run(question)
    #print(last_result)
    return last_result



chat_history = []
st.title("PDK Generator")


def show_history():

    chat_history = st.session_state.chat_history

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message[1]),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True
            )

def main():

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if prompt := st.chat_input("What is up?"):

        if 1==1:




                from langchain.prompts.pipeline import PipelinePromptTemplate

                full_template = """{introduction}
                The problem I need you to solve is:{example}
                Here are some examples:{start}"""
                full_prompt = PromptTemplate.from_template(full_template)
                introduction_template = """ {person}."""
                introduction_prompt = PromptTemplate.from_template(introduction_template)
                example_template = """
                {example_q}
                """
                example_prompt = PromptTemplate.from_template(example_template)
                start_template = """
                {input}
                """
                start_prompt = PromptTemplate.from_template(start_template)
                input_prompts = [
                ("introduction", introduction_prompt),
                ("example", example_prompt),
                ("start", start_prompt)
                ]
                pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
                #pipeline_prompt.input_variables
                #['person', 'example_q', 'input']
                retriever = db.as_retriever(
                   # search_type="similarity_score_threshold",
                 # search_kwargs={"score_threshold": 0.77, "k":1}
                )
                docs = retriever.get_relevant_documents(query=prompt)
                docs_str = ""
                for i in range(len(docs)):
                    page_content = docs[i].page_content + "\n"
                    ocs_str = docs_str + page_content
                chain1 = LLMChain(llm=llm, prompt=pipeline_prompt)

                last_result=load_db1(prompt)
                #chat_history.append((prompt, result))
                with st.spinner('Reading, chunking and embedding file ...'):

                   chat_history.append((prompt, chain1.run(person=last_result, example_q=prompt, input=docs_str)))
                   st.code(chain1.run(person=last_result, example_q=prompt, input=docs_str), language='python')

                st.session_state.messages.append({"role": "assistant", "content": chain1.run(person=last_result, example_q=prompt, input=docs_str)})

                def clear_history():
                  st.session_state.chat_history = []

                if st.session_state.chat_history:
                  st.button("clear clear", on_click=clear_history, use_container_width=True)

        show_history()

if __name__ == '__main__':
    main()
