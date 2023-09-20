import subprocess

from seleniumbase import BaseCase


class PageContentTest(BaseCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_process = subprocess.Popen(["poetry", "run", "streamlit", "run", "app/1_Music_Critique.py"])

    def test_home_page(self) -> None:
        self.open("http://localhost:8501")

        self.assert_title("🎶 Music Critique")

        # Assert the headers
        self.assert_text("🎶 Music Critique")

        # TODO: Complete testing for the main page

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app_process.terminate()
        cls.app_process.wait()