from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from sets import Set
from selenium.webdriver.common.keys import Keys
import io
from time import sleep
import os

class Browser():
  def __init__(self):
    self.driver = webdriver.Chrome('selenium-3.14.1/chromedriver')

  def get(self, url):
    self.driver.get(url)

  def close(self):
    self.driver.quit()

  def get_element_by_css(self, css):
    try:
      element = WebDriverWait(self.driver,10).until(
                  EC.presence_of_element_located((By.CSS_SELECTOR,css))
      )
    except:
      return None
    return element

  def get_element_by_text(self, text):
    try:
      element = WebDriverWait(self.driver,10).until(
                  EC.presence_of_element_located((By.PARTIAL_LINK_TEXT,text))
      )
    except:
      return None
    return element

  def get_elements_by_css(self, css):
    try:
      elements = WebDriverWait(self.driver,10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR,css))
      )
      elements = self.driver.find_elements(By.CSS_SELECTOR,css)
    except:
      return None
    return elements

  def get_attribute(self, element, attr_name):
    return element.get_attribute(attr_name)
  
  def get_attr_value(self, input, attr, type="css"):
    if type == "text":
      element = self.get_element_by_text(input)
    else:
      element = self.get_element_by_css(input)
    if element is None:
      return None
    return self.get_attribute(element, attr)

  def get_attr_values(self, input, attr, type="css"):
    values = Set()
    elements = self.get_elements_by_css(input)
    if elements is None:
      return values
    for element in elements:
      value = self.get_attribute(element, attr)
      if value is not None:
        values.add(str(value))
    return values

  def click_link(self, text):
    try:
      anchor = WebDriverWait(self.driver,10).until(
                EC.presence_of_element_located((By.PARTIAL_LINK_TEXT,text))
      )
      anchor.click()
      return True
    except:
      return False

if __name__ == '__main__':
  booklist = Browser()
  booklist.get('https://openstax.org/subjects')
  textbooks = booklist.get_attr_values('.container .cover a','href') 

  booklist.close()
  browser = Browser()
  
  #textbook = textbooks.pop()


  
  
  for textbook in textbooks:
    browser.get(textbook)
    title = textbook.split('/')
    title = "openstax/"+title[-1]
    if not os.path.exists(title):
      os.mkdir(title)

      main_book = browser.get_attr_value('View online','href','text')
      if main_book is None:
        main_book = browser.get_attr_value('Zobacz w przegl','href','text')
      if main_book is not None:
        browser.get(main_book)  
    
        i = 0
        while (i == 0 or browser.click_link('Next')):
          i+=1
          browser.get_element_by_css('.title-chapter')
          browser.get_element_by_css('[data-type="exercise"]')
          f = io.open(title+"/"+str(i)+".txt","w",encoding="utf-8")
          f.write(browser.get_attr_value('.main-content','innerHTML'))
          f.close()
  