# -*- coding: utf-8 -*-
from .hub.workflow import Workflow
from datetime import timedelta,date
import random,time

__all__ =[
    'Workflow'
]

def egg(seed):
    if seed < 995:
        return 
    try:
        import holidays
    except ImportError:
        return
    cn_holidays = holidays.China()
    today = date.today()
    next_day = today
    while next_day not in cn_holidays:
        next_day += timedelta(days=1)
    if next_day == today:
        print(f"今天是{cn_holidays.get(next_day)}！！！")
    else:
        days_until_next_holiday = (next_day - today).days
        if days_until_next_holiday <=30:
            color = random.randint(31,50)
            # print(f"\033[{color}m {days_until_next_holiday} 天后就要过{cn_holidays.get(next_day)}了！!\033[0m")
            message = f"XEdu小助手提醒您：{days_until_next_holiday} 天后就要过{cn_holidays.get(next_day)}了！!"
            for char in message:
                print(f"\033[{color}m{char}\033[0m", end='', flush=True)
                time.sleep(0.05)
            print()

egg(random.randint(0,1000))
