错误总结:
1.  PackagesNotFoundError: The following packages are not available from current channels
S:  出现以下的原因就是你当前设定的镜像源已经不支持该包了，所以需要重新设定
    这是因为 https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/给了通知：
    “根据 Anaconda 软件源上的说明，Anaconda 和 Miniconda 是 Anaconda, Inc. 的商标，任何未经授权的公开镜像都是不允许的。去年我们曾尝试与公司有关人员联系，但未能取得授权。
    在没有上游授权的情况下，我们无法保证镜像的合法性与服务质量。因此我们决定，在取得授权之前无限期停止 Anaconda 镜像服务。即日起，我们将停止 Anaconda 的更新并隐藏镜像入口链接。
    一个月后，彻底关闭 Anaconda 镜像的文件下载。请现有用户尽快切换至官方下载地址，以免影响正常使用。
    感谢您的理解与支持！
    Update 1 on 2019-05-16: 上述镜像已经被移除”
    也就是说，今天开始必须恢复官方源了
    使用以下语句可以恢复到默认源。
    conda config --remove-key channels

2.  To search for alternate channels that may provide the conda package you’re looking for, navigate to
    https://anaconda.org
    and use the search bar at the top of the page.
S:  根据提示使用默认源 conda config --add channels conda-forge

3.  The following packages are missing from the target environment
    该报错是由于源中缺少相关库导致，说明默认anaconda也无法pip，只能改为手动下载
    pip install python-votesmart

4.  from votesmart import votesmart
S:  在python2下安装python-smart还比较容易，而python3中由于很多函数库的变化
    直接使用python setup.py install 命令来安装的话会导致错误，而导致错误的原因就是python3中没有urllib2,而在votesmart中使用了urllib2函数库，
    所以需要修改votesmart.py文件将其中所有的urllib2库均换成urllib的相应写法，需要修改的地方如下：
    import urllib, urllib2-->import urllib,urllib.request,而要导入urllib.request是要使用其中的urlopen来打开相应的url
    response=urllib2.urlopen(url).read()-->response=urllib.request.urlopen(url).read()
    except urllib2.HTTPError,e:-->except urllib.URLError as e:
    except ValueError,e-->except ValueError as e #注意这里except格式写法的不同

5.  'dict' object has no attribute 'has_key'
S:  Python 3 已弃用 has_key 这一方法
    if not ssCnt.has_key(can): ssCnt[can]=1
    改为if not ssCnt: ssCnt[can]=1

6.  frozenset({3})
S:  错误代码：
    if not  ssCnt: ssCnt[can]=1
    else: ssCnt[can] += 1
    报错的原因是python2.x和3.x的map函数不相同，映射结构不同，返回值也不同
    #2.x
    a = [[1,2],[3,4],[5,6]]
    b = map(frozenset,a)#map返回一个列表
    print b
    #结果为 [frozenset([1, 2]), frozenset([3, 4]), frozenset([5, 6])]
    #3.x
    a = [[1,2],[3,4],[5,6]]
    b = list(map(frozenset,a))#map返回一个可迭代对象
    print(b)
    #结果为 [frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6})]
    应该可以看出两个结果中第二维元素类型不一致
    解决方法为 ：
    if can not in ssCnt: ssCnt[can]=1
    else: ssCnt[can] += 1