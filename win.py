import sys
import traceback
import tkinter
import tkinter.colorchooser
import tkinter.ttk
from tkinter import filedialog

sys.path.append('../ftools')
from ffile import fJson, fTxt
from ftime import fTime

from methods import *


class fWindow(tkinter.Tk):
    def __init__(self, key, title=None, size=None, ico=None):
        super().__init__()
        self.size = None
        self.ico = None
        self.controls = None
        self.vars = None
        self.layout_path = None
        self.method = None
        self._progress_bar_max = None
        self._progress_bar_value = None
        self.init(key, title, size, ico)
        self.mainloop()

    def init(self, key, title, size, ico):
        self.title(title)
        self.size = size
        self.ico = ico
        self.layout_path = None
        self.method = None
        self.controls = dict()
        self.vars = dict()

        if self.size: self.geometry(self.size)
        if self.ico: self.iconbitmap(self.ico)

        if 'method' in manager_data[key].keys(): self.method = manager_data[key]['method']

        if key in manager_data.keys():
            self.layout_path = manager_data[key]['layout_path']
            self.layout()

    def layout(self):
        layout_data = fJson().read(self.layout_path)
        for frame_name, frame_data in layout_data.items():
            frame = tkinter.Frame(self)
            frame.pack()
            row_index = 0
            for line_name, line_data in frame_data.items():
                column_index = 0
                for control_name, control_data in line_data.items():
                    c_type = control_data['type']
                    c_text = control_data['text'] if 'text' in control_data.keys() else None
                    c_width = control_data['width'] if 'width' in control_data.keys() else None
                    c_key = control_data['key'] if 'key' in control_data.keys() else None
                    c_command = control_data['command'] if 'command' in control_data.keys() else None
                    c_value = control_data['value'] if 'value' in control_data.keys() else None
                    c_direction = control_data['direction'] if 'direction' in control_data.keys() else None
                    if c_type == 'Label':
                        control = tkinter.Label(frame, text=c_text, width=c_width)
                    elif c_type == 'Entry':
                        var = tkinter.StringVar()
                        self.vars[c_key] = var
                        control = tkinter.Entry(frame, textvariable=var, width=c_width)
                    elif c_type == 'Button':
                        control = tkinter.Button(frame, text=c_text, width=c_width,
                                                 command=lambda command=c_command, text=c_text, key=c_key:
                                                 self.click(command, text, key))
                    elif c_type == 'Radiobutton':
                        if c_value:
                            control = tkinter.Frame(frame)
                            var = tkinter.IntVar()
                            var.set(0)
                            self.vars[c_key] = var
                            for i, value in enumerate(c_value):
                                control_son = tkinter.Radiobutton(control, text=value, value=i, variable=var)
                                control_son.pack(side=c_direction)
                    elif c_type == 'Checkbutton':
                        var = tkinter.BooleanVar()
                        self.vars[c_key] = var
                        control = tkinter.Checkbutton(frame, text=c_text, variable=var)
                    elif c_type == 'Progressbar':
                        control = tkinter.ttk.Progressbar(frame, length=c_width)
                    elif c_type == 'Message':
                        var = tkinter.StringVar()
                        self.vars[c_key] = var
                        control = tkinter.Message(frame, textvariable=var, width=c_width)
                    else:
                        control = None
                    if c_key: self.controls[c_key] = control
                    control.grid(row=row_index, column=column_index)
                    if c_type == 'Progressbar': control.grid_remove()
                    column_index += 1
                row_index += 1

    def click(self, command, text, key):
        eval(f'{command}(text, key)')

    def new_window(self, title, key):
        """
        打开新窗口
        """
        title, size, ico = init_window_config(key)
        if not size: size = self.size
        if not ico: ico = self.ico
        fToplevel(key, title, size, ico)

    def select_folder(self, text, key):
        folder_path = filedialog.askdirectory()
        if folder_path != '':
            folder_path = folder_path.replace('/', '\\')
            self.vars[key].set(folder_path)
        self.lift()

    def read(self, text, key):
        """
        读取配置
        """
        json_path = filedialog.askopenfilename(parent=self, filetypes=[('.JSON', '.json')])
        if json_path:
            json_data = fJson().read(json_path)
            try:
                for key, value in json_data.items():
                    if type(self.vars[key]) == list:
                        for i, _ in enumerate(self.vars[key]):
                            _.set(value[i])
                    else:
                        self.vars[key].set(value)
            except KeyError:
                self.message('参数过多')
                error(message=f'{fTime().format()}: {self.read.__name__}\n{traceback.format_exc()}')

    def readin(self, text, key):
        """
        写入配置
        """
        json_path = filedialog.asksaveasfilename(parent=self, filetypes=[('.JSON', '.json')], defaultextension='.json')
        if json_path: fJson().readin(json_path, self.formatting_config(True), indent=4)

    def confirm(self, text, key):
        """
        遍历文件处理
        """
        self.controls['confirm']['state'] = tkinter.DISABLED
        try:
            eval(f'{self.method}(self, self.formatting_config())')
        except Exception:
            self.message('参数错误')
            error(message=f'{fTime().format()}: {self.confirm.__name__}\n{traceback.format_exc()}')
        self.controls['confirm']['state'] = tkinter.NORMAL

    def select_color(self, text, key):
        """
        选择颜色
        """
        r = tkinter.colorchooser.askcolor(title='选择颜色')
        self.vars['mask_color'].set(str(r[0]))
        self.lift()

    def message(self, text: str = ''):
        tkinter.messagebox.showerror(title=text, message=text)
        self.lift()

    @property
    def progress_bar_value(self):
        return self._progress_bar_value

    @progress_bar_value.setter
    def progress_bar_value(self, value: int = 0):
        self.controls['progress_bar']['value'] = value
        self.update()

    @progress_bar_value.getter
    def progress_bar_value(self):
        return self.controls['progress_bar']['value']

    @property
    def progress_bar_max(self):
        return self._progress_bar_max

    @progress_bar_max.setter
    def progress_bar_max(self, value: int):
        self.controls['progress_bar'].grid()
        self.controls['progress_bar']['maximum'] = value
        self.progress_bar_value = 0

    def formatting_config(self, is_save: bool = False):
        """
        格式化配置
        """
        config = dict()
        try:
            for key, value in self.vars.items():
                if type(value) == list:
                    value = [_.get() for _ in value]
                elif type(value) == bool:
                    pass
                else:
                    value = value.get()
                    if type(value) == str:
                        if self.is_int(value):
                            value = int(value)
                        elif self.is_float(value):
                            value = float(value)
                        elif self.is_rgb(value):
                            value = [int(_) for _ in value[1:-1].replace(' ', '').split(',')]
                            value = (value[2], value[1], value[0])
                        else:
                            pass
                    else:
                        pass
                if is_save:
                    if type(value) == tuple: value = str(value)
                config[key] = value
            return config
        except Exception:
            self.message('参数初始化错误')
            error(message=f'{fTime().format()}: {self.formatting_config.__name__}\n{traceback.format_exc()}')

    def is_int(self, text: str):
        if text.find('.') == -1:
            if text.isdigit():
                return True
            elif len(text) > 1 and text[0] == '-' and text[1:].isdigit():
                return True
        return False

    def is_float(self, text: str):
        text = text.split('.')
        if len(text) == 2 and text[1].isdigit() and (self.is_int(text[0]) or text[0] == ''):
            return True
        else:
            return False

    def is_rgb(self, text: str):
        if len(text) > 1 and text[0] == '(' and text[-1] == ')':
            text = text[1:-1].replace(' ', '').split(',')
            if len(text) == 3 and text[0].isdigit() and text[1].isdigit() and text[2].isdigit():
                return True
        return False


class fToplevel(tkinter.Toplevel, fWindow):
    def __init__(self, key, title, size=None, ico=None):
        super().__init__()
        self.size = None
        self.ico = None
        self.controls = None
        self.stringvars = None
        self.layout_path = None
        self.method = None
        self._progress_bar_max = None
        self._progress_bar_value = None
        fWindow.init(self, key, title, size, ico)


def train_platform_simplify_paths(self, config):
    self.vars['out_path'].set(config['scan_path'].replace(',', '\n'))


def init_window_config(key):
    title = manager_data[key]['text'] if 'text' in manager_data[key].keys() else None
    size = manager_data[key]['size'] if 'size' in manager_data[key].keys() else None
    ico = manager_data[key]['ico'] if 'ico' in manager_data[key].keys() else None
    return title, size, ico


def error(path=f'./error/{fTime().date()}.txt', message='\n'):
    fTxt().add(path, message)


if __name__ == '__main__':
    pass

    try:
        manager_path = './data/manager.json'
        manager_data = fJson().read(manager_path)
        main_window_key = list(manager_data.keys())[0]

        main_title, main_size, main_ico = init_window_config(main_window_key)
        main_window = fWindow(main_window_key, main_title, main_size, main_ico)
    except Exception:
        error(message=f'{fTime().format()}: {__name__}\n{traceback.format_exc()}')
