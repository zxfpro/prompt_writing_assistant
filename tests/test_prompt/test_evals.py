from prompt_writing_assistant.prompt_helper import Base_Evals

from dotenv import load_dotenv
load_dotenv(".env", override=True)

# Evals

class AEvals(Base_Evals):
    def __init__(self):
        self.MIN_SUCCESS_RATE = 00.0 # 这里定义通过阈值, 高于该比例则通过

    def _assert_eval_function(self,params):

        # prompt running
        assert params[0] + params[1] == params[-1]
        # self.person_evals(params[:-1],params[-1],params[-1])
        # self.rule_evals(params[-1],params[-1])
        print(params,'params111')

    def rule_evals(self):
        return super().rule_evals()
    
    def global_evals(self):
        pass

    def local_evals(self):
        pass
        

def test_evals():
    aeval = AEvals()
    test_cases = [
            (1, 2, 3),      # Pass
            (-1, 1, 0),     # Pass
            (0, 0, 0),      # Pass
            (100, 200, None), # Fail
            (-5, -3, -8)    # Pass
        ]
    aeval.get_success_rate(test_cases)


def test_evals_for_auto():
    aeval = AEvals()
    test_cases = [
            (1, 2, 3),      # Pass
            (-1, 1, 0),     # Pass
            (0, 0, 0),      # Pass
            (100, 200, None), # Fail
            (-5, -3, -8)    # Pass
        ]
    result = aeval.get_success_rate_for_auto(test_cases)
    print(result,'result')

# 一个小环节 就要定义一个评价类

class BEvals(Base_Evals):
    def __init__(self):
        self.MIN_SUCCESS_RATE = 80.0 # 这里定义通过阈值, 高于该比例则通过

    def _assert_eval_function(self,params):

        # prompt running
        assert params[0] + params[1] == params[-1]
        # self.person_evals(params[:-1],params[-1],params[-1])
        # self.rule_evals(params[-1],params[-1])
        print(params,'params111')

    def rule_evals(self):
        return super().rule_evals()
    
    def global_evals(self):
        pass

    def local_evals(self):
        pass
        

def test_evals_for_auto2():
    from prompt_writing_assistant.prompt_helper import Intel,IntellectType
    from prompt_writing_assistant.utils import create_session
    from prompt_writing_assistant.database import UseCase
    intels = Intel()

    with create_session(intels.engine) as session:
        result = session.query(UseCase).filter(
            Prompt.prompt_id == target_prompt_id
        ).order_by(
            Prompt.timestamp.desc(),
            Prompt.version.desc()
        ).first()


    db_manager = MySQLManager()
    table_name = "use_case"
    
    user_by_id_1 = db_manager.select(table_name, conditions="prompt_id = %s", params=("db_help_001",), fetch_all=False)
    if user_by_id_1:
        print(user_by_id_1)



    # beval = BEvals()
    # # 获取prompt_id 对应的数据use_case
    # # 2 对这些做人工标注和审查
    # # 3 获取并组织成如下结构
    # test_cases = [
    #         (1, 2, 3),      # Pass
    #         (-1, 1, 0),     # Pass
    #         (0, 0, 0),      # Pass
    #         (100, 200, None), # Fail
    #         (-5, -3, -8)    # Pass
    #     ]
    # result = beval.get_success_rate_for_auto(test_cases)
    # print(result,'result')
