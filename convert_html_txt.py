def convert_html_to_txt(html_file, txt_file):
    from bs4 import BeautifulSoup

    with open(html_file, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
        # remove extra newlines
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    # remove extra whitespace
    text = text.strip()
    with open(txt_file, "w", encoding="utf-8") as file:
        file.write(text)


if __name__ == "__main__":
    convert_html_to_txt("sun256.html", "sun256.txt")
