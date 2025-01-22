# coding: utf8
import os
import sys
from time import sleep
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import datetime


class InstagramScraper:
    def __init__(self, username, password, driver_path):
        self.username = username
        self.password = password
        self.driver_path = driver_path
        self.driver = None
        self.comment_index = 2300  # 初始化留言編號

    def initialize_driver(self):
        options = webdriver.ChromeOptions()
        self.driver = webdriver.Chrome(service=Service(self.driver_path), options=options)

    def login(self):
        print("Logging into Instagram...")
        self.driver.get("https://www.instagram.com/accounts/login/")
        sleep(2.5)  # 等待頁面載入

        # 輸入帳號
        username_input = self.driver.find_element(By.NAME, "username")
        username_input.send_keys(self.username)
        print("Entered username.")

        # 輸入密碼
        password_input = self.driver.find_element(By.NAME, "password")
        password_input.send_keys(self.password)
        print("Entered password.")

        # 點擊登入按鈕
        login_button = self.driver.find_element(By.XPATH, '//*[@id="loginForm"]/div/div[3]/button')
        login_button.click()
        print("Clicked login button.")

        sleep(5)  # 等待登入完成
        print("Logged in successfully.")

    def scrape_post(self, url):
        print("Navigating to the post URL...")
        self.driver.get(url)
        sleep(6)  # 等待頁面載入
        
        # 取得頁面 HTML
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # 抓取貼文標題
        try:
            title = soup.find("meta", property="og:title")["content"]
            print("Post Title:", title)
        except (TypeError, KeyError):
            print("Unable to fetch post title.")
            title = None

        # 抓取留言內容
        print("Fetching comments...")

        # 抓取留言內容
        comments = []
        self.load_all_comments()  # 呼叫新增的滾動與加載邏輯

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        comment_elements = soup.find_all("span", class_="x1lliihq x1plvlek xryxfnj x1n2onr6 x1ji0vk5 x18bv5gf x193iq5w xeuugli x1fj9vlw x13faqbe x1vvkbs x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1i0vuye xvs91rp xo1l8bm x5n08af x10wh9bi x1wdrske x8viiok x18hxmgj")
        for element in comment_elements:
            comments.append(element.text)

        print(f"Fetched {len(comments)} comments.")
        return {"title": title, "comments": comments}
    
    def load_all_comments(self):
        # 用 BeautifulSoup 檢查元素是否存在
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        scrollable_element_check = soup.find("div", class_="x5yr21d xw2csxc x1odjw0f x1n2onr6")
        if scrollable_element_check:
            print("Scrollable element found with BeautifulSoup!")

        # 等待該元素存在
        try:
            scrollable_element = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.x5yr21d.xw2csxc.x1odjw0f.x1n2onr6"))
            )
            print("Scrollable element found with Selenium!")
        except TimeoutException:
            print("Failed to locate the scrollable element.")
            self.driver.quit()

        # 滾動並等待載入更多留言
        scroll_count = 0  # 計數滾動次數
        previous_comments_count = 0  # 記錄上次滾動後的留言數量
        max_scrolls = 7  # 設定最多滾動次數

        while scroll_count < max_scrolls:
            # 等待新留言加載
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            comment_elements = soup.find_all("span", class_="x1lliihq x1plvlek xryxfnj x1n2onr6 x1ji0vk5 x18bv5gf x193iq5w xeuugli x1fj9vlw x13faqbe x1vvkbs x1s928wv xhkezso x1gmr53x x1cpjm7i x1fgarty x1943h6x x1i0vuye xvs91rp xo1l8bm x5n08af x10wh9bi x1wdrske x8viiok x18hxmgj")
            new_comments_count = len(comment_elements)

            if new_comments_count > previous_comments_count:
                print(f"Found {new_comments_count - previous_comments_count} new comments.")
                previous_comments_count = new_comments_count
            else:
                print("No new comments found, stopping scroll.")
                break  # 如果沒有新留言就停止滾動

            # 滾動容器
            self.driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_element)
            # 延長等待時間，確保留言有足夠時間加載
            sleep(5)  # 等待頁面載入更多留言，增加等待時間
            scroll_count += 1
            print(f"Scrolled {scroll_count}/{max_scrolls} times.")            

    def close_driver(self):
        if self.driver:
            self.driver.quit()

    def save_comments_individually(self, comments, base_path):
        for i, comment in enumerate(comments):
            if i % 2 == 0:  # 只保存奇數索引的留言
                file_path = os.path.join(base_path, f"comment_{self.comment_index}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(comment)
                self.comment_index += 1

if __name__ == "__main__":
    # 提供您的 Instagram 帳號和密碼
    INSTAGRAM_USERNAME = "cchh89317"
    INSTAGRAM_PASSWORD = "damn0123"
    # INSTAGRAM_USERNAME = "textproject00"
    # INSTAGRAM_PASSWORD = "fuck11225"
    CHROME_DRIVER_PATH = r"/usr/local/bin/chromedriver"  # 為本地 chromedriver 路徑

    # 從 urls.txt 檔案中讀取網址
    with open("./_爬蟲/urls.txt", "r") as file:
        urls = file.readlines()

    scraper = InstagramScraper(INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD, CHROME_DRIVER_PATH)
    scraper.initialize_driver()

    try:
        scraper.login()
        for i, url in enumerate(urls):
            url = url.strip()  # 去除換行符號
            if url:
                post_data = scraper.scrape_post(url)
                print("Scraped Data:", post_data)
                
                # 生成唯一的檔案名稱 用日期時間
                nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                output_dir = f"./_爬蟲/Result"
                os.makedirs(output_dir, exist_ok=True)

                # 保存每個留言到獨立的 .txt 檔案
                scraper.save_comments_individually(post_data["comments"], output_dir)
    except Exception as e:
        print("Error occurred:", e)
    finally:
        scraper.close_driver()