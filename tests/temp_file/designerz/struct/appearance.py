'''外观模式'''

class Appearance():
    """
使用效果就是直接调用老接口的对象的话对方咩有这个方法,
那么就将老接口传入一个适配器,这样,老接口就有了新方法 有点像电源适配器,或者插头转换器
## 适配器模式

    """
    def __init__(self):
        self.p = """
class TV:
    def on(self):
        print("TV is on")

    def off(self):
        print("TV is off")

class SoundSystem:
    def on(self):
        print("Sound system is on")

    def off(self):
        print("Sound system is off")

    def set_volume(self, volume):
        print(f"Sound system volume set to {volume}")

class DVDPlayer:
    def on(self):
        print("DVD player is on")

    def off(self):
        print("DVD player is off")

    def play(self, movie):
        print(f"Playing movie: {movie}")

# 外观类
class HomeTheaterFacade:
    def __init__(self, tv: TV, sound_system: SoundSystem, dvd_player: DVDPlayer):
        self._tv = tv
        self._sound_system = sound_system
        self._dvd_player = dvd_player

    def watch_movie(self, movie):
        print("Get ready to watch a movie...")
        self._tv.on()
        self._sound_system.on()
        self._sound_system.set_volume(20)
        self._dvd_player.on()
        self._dvd_player.play(movie)

    def end_movie(self):
        print("Shutting down the home theater...")
        self._tv.off()
        self._sound_system.off()
        self._dvd_player.off()

# 使用外观模式
tv = TV()
sound_system = SoundSystem()
dvd_player = DVDPlayer()

home_theater = HomeTheaterFacade(tv, sound_system, dvd_player)
home_theater.watch_movie("Inception")
home_theater.end_movie()

## 外观模式其实就是常用的综合类嘛
# main的类别 work的工作  自动化工作流的想法
# 外观模式就是  快捷指令
"""
    def __repr__(self):
        return self.p