import sys
try:
    from .ast_v2_app import App
except ImportError:
    try:
        from ast_v2_app import App
    except ModuleNotFoundError as e:
        from src.cell.montage.AST_v3.ast_v2_app import App

def launch_gui(dir_to_process: str = None) -> bool:
    """
    Runs the AST V2 GUI

    :param dir_to_process: directory to montage image, defaults to None
    :type dir_to_process: str, optional
    :return: whether the montaged finished properly or not
    :rtype: bool
    """
    import logging
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # try:
    app = App(dir_to_process)
    # app.state('zoomed')
    app.mainloop()
        # return True
    # except:
    #     return False

# Executable of the main GUI function to place the unconnected components
if __name__ == '__main__':
    launch_gui()
