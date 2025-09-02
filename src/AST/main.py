try:
    from .ast_app import App
except ImportError:
    try:
        from ast_app import App
    except ModuleNotFoundError:
        import sys, os
        from pathlib import Path
        sys.path.append(os.path.abspath(Path(os.getcwd()).parent))
        try:
            from ast_app import App
        except ModuleNotFoundError:
            from AST.ast_app import App

def main() -> None:
    app = App()
    app.mainloop()
    # pyinstaller -F --hidden-import=tkinter --hidden-import=tkinter.filedialog --hidden-import=numpy --hidden-import=PIL main.py
    # https://stackoverflow.com/questions/41228933/importerror-no-module-named-tkinter-after-pyinstaller
    # all second-level imports


if __name__ == '__main__':
    main()
