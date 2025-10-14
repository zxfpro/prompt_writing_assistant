""" # 观察者模式 """
class Watcher():
    def __init__(self):
        self.p = """
from abc import ABC, abstractmethod
from typing import List

# 主题接口
class Subject(ABC):
    @abstractmethod
    def attach(self, observer: 'Observer'):
        pass

    @abstractmethod
    def detach(self, observer: 'Observer'):
        pass

    @abstractmethod
    def notify(self):
        pass

# 具体主题
class WeatherData(Subject):
    '''
    weather_data = WeatherData()
    print('################')
    temp_display1 = TemperatureDisplay("Main")
    temp_display2 = TemperatureDisplay("Secondary")
    print('################')
    weather_data.attach(temp_display1)
    weather_data.attach(temp_display2)
    print('################')
    weather_data.set_temperature(25.0)
    weather_data.set_temperature(30.0)
    print('################')
    weather_data.detach(temp_display1)
    weather_data.set_temperature(35.0)
    '''
    def __init__(self):
        self._observers: List[Observer] = []
        self._temperature: float = 0.0

    def attach(self, observer: 'Observer'):# 加载订阅者
        self._observers.append(observer)

    def detach(self, observer: 'Observer'):# 卸载订阅者
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self._temperature)

    def set_temperature(self, temperature: float):
        print(f"WeatherData: setting temperature to {temperature}")
        self._temperature = temperature
        self.notify()

# 观察者接口
class Observer(ABC):
    @abstractmethod
    def update(self, temperature: float):
        pass

# 具体观察者
class TemperatureDisplay(Observer): #订阅者
    def __init__(self, name: str):
        self._name = name
        self._temperature = 0.0

    def update(self, temperature: float):
        self._temperature = temperature
        print(f"{self._name} Display: temperature updated to {self._temperature}")

"""
    
    def __repr__(self):
        return self.p