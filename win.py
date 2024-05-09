import sys
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
        """
        初始化布局
        """
        layout_data = fJson().read(self.layout_path)
        for frame_name, frame_data in layout_data.items():
            frame = tkinter.Frame(self)
            frame.pack()
            row_index = 0
            for line_name, line_data in frame_data.items():
                column_index = 0
                for control_name, control_data in line_data.items():
                    c, keys = dict(), control_data.keys()
                    c['columnspan'] = control_data['columnspan'] if 'columnspan' in keys else 1
                    c['rowspan'] = control_data['rowspan'] if 'rowspan' in keys else 1
                    c['type'] = control_data['type'] if 'type' in keys else 'Label'  # 类型
                    c['text'] = control_data['text'] if 'text' in keys else None  # 文本
                    c['width'] = control_data['width'] if 'width' in keys else None  # 宽度
                    c['key'] = control_data['key'] if 'key' in keys else None  # 主键
                    c['command'] = control_data['command'] if 'command' in keys else None  # 按键命令
                    c['value'] = control_data['value'] if 'value' in keys else None  # 预置值
                    c['current'] = control_data['current'] if 'current' in keys else 0  # 首选预置值
                    c['direction'] = control_data['direction'] if 'direction' in keys else None  # 单选框选项方向
                    c['filetypes'] = control_data['filetypes'] if 'filetypes' in keys else []  # 文件选择后缀名
                    c['lift'] = control_data['lift'] if 'lift' in keys else None  # 点击任务结束后是否置顶
                    c['sticky'] = control_data['sticky'] if 'sticky' in keys else ''  # 对齐
                    c['bg'] = control_data['bg'] if 'bg' in keys else None
                    c['fg'] = control_data['fg'] if 'fg' in keys else None

                    if c['type'] == 'Label':
                        control = tkinter.Label(frame, text=c['text'], width=c['width'])
                    elif c['type'] == 'Entry':
                        var = tkinter.StringVar()
                        self.vars[c['key']] = var
                        control = tkinter.Entry(frame, textvariable=var, width=c['width'])
                    elif c['type'] == 'Button':
                        control = tkinter.Button(frame, text=c['text'], width=c['width'],
                                                 command=lambda command=c['command'], config=c: self.click(command,
                                                                                                           config),
                                                 fg=c['fg'])
                    elif c['type'] == 'Radiobutton':
                        control = tkinter.Frame(frame)
                        var = tkinter.IntVar()
                        var.set(0)
                        self.vars[c['key']] = var
                        for i, value in enumerate(c['value']):
                            control_son = tkinter.Radiobutton(control, text=value, value=i, variable=var)
                            control_son.pack(side=c['direction'])
                    elif c['type'] == 'Checkbutton':
                        var = tkinter.BooleanVar()
                        self.vars[c['key']] = var
                        control = tkinter.Checkbutton(frame, text=c['text'], width=c['width'], variable=var)
                    elif c['type'] == 'Combobox':
                        var = tkinter.StringVar()
                        self.vars[c['key']] = var
                        control = tkinter.ttk.Combobox(frame, textvariable=var, width=c['width'], values=c['value'])
                        control.current(c['current'])
                    elif c['type'] == 'Progressbar':
                        control = tkinter.ttk.Progressbar(frame, length=c['width'])
                        if c['value'] == 'loop':
                            control['mode'] = 'indeterminate'
                            control['orient'] = tkinter.HORIZONTAL
                            control.start(c['current'])
                    elif c['type'] == 'Message':
                        var = tkinter.StringVar()
                        self.vars[c['key']] = var
                        control = tkinter.Message(frame, textvariable=var, width=c['width'])
                    else:
                        control = None
                    if c['key']: self.controls[c['key']] = control
                    control.grid(row=row_index, column=column_index, rowspan=c['rowspan'], columnspan=c['columnspan'],
                                 sticky=c['sticky'])
                    if c['type'] == 'Progressbar': control.grid_remove()
                    column_index += c['columnspan']
                row_index += c['rowspan']

    def click(self, command, c):
        eval(f'{command}(c)')
        if c['lift']: self.lift()

    def new_window(self, c):
        """
        打开新窗口
        """
        title, size, ico = init_window_config(c['key'])
        if not size: size = self.size
        if not ico: ico = self.ico
        fToplevel(c['key'], title, size, ico)

    def message(self, text: str = '', message_type: str = 'error'):
        message_types = ['info', 'warning', 'error']
        message_shows = [tkinter.messagebox.showinfo, tkinter.messagebox.showwarning, tkinter.messagebox.showerror]
        message_info = ['提示', '警告', '错误']
        message_type_index = message_types.index(message_type)
        message_shows[message_type_index](title=message_info[message_type_index], message=text)
        self.lift()

    def confirm(self, c):
        """
        遍历文件处理
        """
        self.controls['confirm']['state'] = tkinter.DISABLED
        try:
            eval(f'{self.method}(self, self.formatting_config())')
        except Exception:
            self.message('执行失败')
            error(message=f'{fTime().format()}: {self.confirm.__name__}\n{traceback.format_exc()}')
        self.controls['confirm']['state'] = tkinter.NORMAL

    def ask_folder(self):
        folder_path = filedialog.askdirectory(parent=self)
        return folder_path

    def ask_file(self, filetypes):
        file_path = filedialog.askopenfilename(parent=self, filetypes=filetypes)
        return file_path

    def select_folder(self, c):
        folder_path = self.ask_folder()
        if folder_path != '':
            # folder_path = folder_path.replace('/', '\\')
            self.vars[c['key']].set(folder_path)

    def select_file(self, c):
        file_path = self.ask_file(c['filetypes'])
        if file_path != '':
            # file_path = file_path.replace('/', '\\')
            self.vars[c['key']].set(file_path)

    def select_color(self, text, key):
        """
        选择颜色
        """
        r = tkinter.colorchooser.askcolor(title='选择颜色')
        self.vars['mask_color'].set(str(r[0]))

    def read_config(self, c):
        """
        读取配置
        """
        json_path = self.ask_file([['.json', '.JSON']])
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
                self.message('参数读取失败')
                error(message=f'{fTime().format()}: {self.read.__name__}\n{traceback.format_exc()}')

    def readin_config(self, c):
        """
        写入配置
        """
        json_path = filedialog.asksaveasfilename(parent=self, filetypes=[('.JSON', '.json')], defaultextension='.json')
        if json_path: fJson().readin(json_path, self.formatting_config(True), indent=4)

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
        self.progress_bar_show = True
        self.controls['progress_bar']['maximum'] = value
        self.progress_bar_value = 0

    @property
    def progress_bar_show(self):
        return self.progress_bar_show

    @progress_bar_show.setter
    def progress_bar_show(self, show: bool = True):
        if show:
            self.controls['progress_bar'].grid()
        else:
            self.controls['progress_bar'].grid_remove()

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
            self.message('参数初始化失败')
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


def error(path=f'./error/{fTime().format()}.txt', message='\n'):
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
