# utils/display.py
def print_section(title: str):
    print(f"\n{'=' * 80}")
    print(f"{title.center(80)}")
    print(f"{'=' * 80}")

def print_subsection(title: str):
    print(f"\n{'-' * 40} {title} {'-' * 40}")