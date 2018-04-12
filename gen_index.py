# -*- coding:utf-8 -*-

import os
import re
import platform

# 需要忽略的文件或者文件夹的名字
ignore_names = ['.git', 'README.md']

def CheckIgnore(path):
    abs_path = os.path.abspath(path)
    for n in ignore_names:
        name = os.path.split(abs_path)[1]
        if name == n:
            return True
    return False

# 符合建立所以要求的文件的后缀名
ext_names = ['md', 'markdown']

def checkExtension(name):
    ext = os.path.splitext(name)[1]
    if len(ext) == 0:
        return False
    ext = ext[1:]
    for n in ext_names:
        if ext == n:
            return True
    return False

class FileNode(object):

    def __init__(self, path, level = 1):
        self.path = path
        self.level = level
        self.is_dir = os.path.isdir(path)
        self.abs_path = os.path.abspath(self.path)
        self.name = os.path.split(self.abs_path)[1]
        self.children = []

    def pharseDir(self):
        if self.is_dir:
            for lists in os.listdir(self.path):
                if CheckIgnore(os.path.join(self.path, lists)):
                    continue
                child_node = FileNode(os.path.join(self.path, lists), self.level + 1)
                child_node.pharseDir()
                self.children.append(child_node)

    def printAll(self, abs_path = False):
        if abs_path:
            print('%s : %d' % (self.abs_path, self.level))
        else:
            print('%s : %d' % (self.path, self.level))
        for child in self.children:
            child.printAll(abs_path)

    def isDir(self):
        return self.is_dir

class IndexWriter(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.is_opened = False
        self.add_line = False
        self.openFile()

    def openFile(self):
        if self.is_opened is True:
            return
        if os.path.isdir(self.file_name):
            raise ValueError('%s is not a file!' % self.file_name)
        self.file_handler = open(self.file_name, 'wt')
        self.is_opened = True

    def closeFile(self):
        if self.is_opened is False:
            return
        self.file_handler.close()
        self.is_opened = False

    def writeStr(self, s):
        if platform.system() == 'Windows':
            s = s.decode('gbk').encode('utf-8')
        self.file_handler.write(s)

    def writeTitle(self, name, level):
        if self.add_line:
            self.writeStr('\n')
            self.add_line = False
        self.writeStr('#' * level + ' ')
        self.writeStr('%s\n\n' % name)

    def writeItem(self, name ,path):
        path = self.convertPathSpliter(path)
        self.writeStr('- [%s](%s)\n' % (name, path))
        self.add_line = True

    def convertPathSpliter(self, path):
        return '/'.join(path.split(os.path.sep))

class IndexGenerator(object):

    def __init__(self, path):
        self.path = path
        self.file_node = FileNode(self.path)
        self.file_node.pharseDir()
        self.have_item_dict = {}
        self.index_writer = None

    def generate(self, index_file_name):
        self.index_file_name = index_file_name
        self.index_writer = IndexWriter(self.index_file_name)

        self.recurseGenerate(self.file_node)

    def recurseGenerate(self, file_node):
        if self.index_writer is None:
            raise IOError('Index Writer has not been initialized yet.')

        if file_node.isDir():
            if self.checkHaveItem(file_node):
                self.index_writer.writeTitle(file_node.name, file_node.level)
                # Handle with file first
                for child in file_node.children:
                    if not child.isDir():
                        self.recurseGenerate(child)
                for child in file_node.children:
                    if child.isDir():
                        self.recurseGenerate(child)
        else:
            if self.checkHaveItem(file_node):
                item_name = os.path.splitext(file_node.name)[0]
                self.index_writer.writeItem(item_name, file_node.path)

    def checkHaveItem(self, file_node):
        # Check the dict, if exists, return the dict result
        if self.have_item_dict.has_key(file_node.abs_path):
            return self.have_item_dict[file_node.abs_path]
        # Check if this file has the correct extension.
        # If have, return True
        if checkExtension(file_node.name):
            self.have_item_dict[file_node.abs_path] = True
            return True
        # Check all its child node recursivly
        for child in file_node.children:
            if self.checkHaveItem(child):
                self.have_item_dict[file_node.abs_path] = True
                return True
        # IF all the check before fails,
        # then no item is in this file node,
        # return False
        self.have_item_dict[file_node.abs_path] = False
        return False

def main():
    generator = IndexGenerator('.')
    generator.generate('README.md')

if __name__ == '__main__':
    main()
