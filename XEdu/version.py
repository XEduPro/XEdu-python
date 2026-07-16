import os

__version__='2.1.0'
__path__=os.path.abspath(os.getcwd())

def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            version_info.append(int(patch_version[0]))
            version_info.append(f'rc{patch_version[1]}')
    return tuple(version_info)

def hello():
                                                 
    print("""                                             
                                                       dddddddd                  
XXXXXXX       XXXXXXXEEEEEEEEEEEEEEEEEEEEEE            d::::::d                  
X:::::X       X:::::XE::::::::::::::::::::E            d::::::d                  
X:::::X       X:::::XE::::::::::::::::::::E            d::::::d                  
X::::::X     X::::::XEE::::::EEEEEEEEE::::E            d:::::d                   
XXX:::::X   X:::::XXX  E:::::E       EEEEEE    ddddddddd:::::d uuuuuu    uuuuuu  
   X:::::X X:::::X     E:::::E               dd::::::::::::::d u::::u    u::::u  
    X:::::X:::::X      E::::::EEEEEEEEEE    d::::::::::::::::d u::::u    u::::u  
     X:::::::::X       E:::::::::::::::E   d:::::::ddddd:::::d u::::u    u::::u  
     X:::::::::X       E:::::::::::::::E   d::::::d    d:::::d u::::u    u::::u  
    X:::::X:::::X      E::::::EEEEEEEEEE   d:::::d     d:::::d u::::u    u::::u  
   X:::::X X:::::X     E:::::E             d:::::d     d:::::d u::::u    u::::u  
XXX:::::X   X:::::XXX  E:::::E       EEEEEEd:::::d     d:::::d u:::::uuuu:::::u  
X::::::X     X::::::XEE::::::EEEEEEEE:::::Ed::::::ddddd::::::ddu:::::::::::::::uu
X:::::X       X:::::XE::::::::::::::::::::E d:::::::::::::::::d u:::::::::::::::u
X:::::X       X:::::XE::::::::::::::::::::E  d:::::::::ddd::::d  uu::::::::uu:::u
XXXXXXX       XXXXXXXEEEEEEEEEEEEEEEEEEEEEE   ddddddddd   ddddd    uuuuuuuu  uuuu
    """)
    print("相关网址：")
    print("-文档网址 :  https://xedu.readthedocs.io")
    print("-官网网址 :  https://www.openinnolab.org.cn/pjedu/xedu/mmedu")

version_info = parse_version_info(__version__)
# path_info = parse_version_info(__path__)
