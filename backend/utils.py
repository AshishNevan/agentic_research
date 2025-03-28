from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

static_url = os.getenv("FASTAPI_URL", "http://localhost:8000") + "/static"


def move_static_files():
    static_dir = Path("static")
    static_dir.mkdir(parents=True, exist_ok=True)
    home_dir = Path("./")
    png_files = [f for f in home_dir.iterdir() if f.suffix == ".png"]

    for png_file in png_files:
        src_path = home_dir / png_file
        dst_path = static_dir / png_file
        if not dst_path.exists():
            src_path.rename(dst_path)


def exec_viz_code(viz_code: str):
    try:
        exec(viz_code)
        return True
    except Exception as e:
        print(f"Error executing visualization code: {e}")
        return False


def append_viz_to_report(report: str):
    static_dir = Path("static")
    if static_dir.exists() and any(static_dir.iterdir()):
        report += "\n\n## Generated Charts\n"
        for png_file in static_dir.iterdir():
            if png_file.suffix == ".png":
                report += f"\n![{png_file.stem}]({static_url}/{png_file.name})"
    return report
