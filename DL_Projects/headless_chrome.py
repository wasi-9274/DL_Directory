from selenium import webdriver
from time import sleep

driver = webdriver.Chrome('/home/wasi/Downloads/unsplash wallpapers/chromedriver_linux64/chromedriver')
driver.get("https://www.google.com")
print(driver.title)
sleep(5)
driver.close()