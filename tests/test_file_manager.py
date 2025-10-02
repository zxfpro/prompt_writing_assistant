from prompt_writing_assistant.file_manager import ContentManager, TextType

from dotenv import load_dotenv
load_dotenv(".env", override=True)
import pytest

# 内容池
class Test_ContentManager():

    @pytest.fixture
    def content_manager(self):
        return ContentManager()
    
    def test_save_content(self, content_manager):
        result =content_manager.save_content(
            name = "yaml_文件变动规律",
            type = TextType.Code,
            text = '''

    
''',
        )
        print(result,'result')

    def test_search(self, content_manager):
        result = content_manager.search(
            name = "装饰器-struct",
        )
        print(result,'result')

    def test_similarity(self,content_manager):
        result = content_manager.similarity(
            content = "每日资讯还是要具备的, 你要知道的内容和111",
            limit = 2
        )
        print(result,'result')
