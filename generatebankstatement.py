import random
import os
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

BANK_NAMES = [
    "Bank of Scotland",
    "Lloyds Bank",
    "Barclays",
    "NatWest",
    "HSBC UK",
]

STREET_NAMES = [
    "Cloughs Road",
    "Baker Street",
    "High Street",
    "King's Road",
    "Oak Avenue",
    "Station Road",
]

CITIES = [
    "Glasgow",
    "London",
    "Edinburgh",
    "Manchester",
    "Birmingham",
]

SURNAMES = ["Smith", "Johnson", "Brown", "Miller", "Taylor", "Wilson"]
FIRST_NAMES = ["John", "Emma", "Olivia", "Liam", "Noah", "Sophia", "James"]

MERCHANTS = [
    "TESCO STORES",
    "SAINSBURY'S",
    "AMAZON UK",
    "UBER TRIP",
    "DELIVEROO",
    "PRET A MANGER",
    "MCDONALD'S",
    "TRAINLINE",
]


def random_sort_code():
    return f"{random.randint(10,99)}-{random.randint(10,99)}-{random.randint(10,99)}"


def random_account_number():
    return f"{random.randint(10_000_000, 99_999_999)}"


def random_iban_gb(sort_code, account_number):
    # Very rough fake IBAN, not a real checksum
    bban = f"BOFS{sort_code.replace('-', '')}{account_number}"
    return "GB" + str(random.randint(10, 99)) + " " + " ".join(
        [bban[i : i + 4] for i in range(0, len(bban), 4)]
    )


def random_postcode():
    # Simple UK-ish postcode
    area = random.choice(["G1", "SW1", "E14", "M4", "B1"])
    district = random.randint(1, 9)
    suffix = random.choice(["AA", "AB", "BA", "BB", "XZ"])
    return f"{area}{district} {random.randint(1, 9)}{suffix}"


def random_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(SURNAMES)}"


def random_address():
    line1 = f"{random.randint(1, 99)} {random.choice(STREET_NAMES)}"
    city = random.choice(CITIES)
    postcode = random_postcode()
    return line1, city, postcode


def generate_transactions(start_balance=1000.00, days=30, max_tx_per_day=3):
    txs = []
    current_date = datetime.today() - timedelta(days=days)
    balance = start_balance

    for day in range(days):
        date = current_date + timedelta(days=day)
        for _ in range(random.randint(0, max_tx_per_day)):
            desc = random.choice(MERCHANTS)
            amount = round(random.uniform(-150.0, 200.0), 2)
            balance += amount
            txs.append(
                {
                    "date": date.strftime("%d/%m/%Y"),
                    "desc": desc,
                    "amount": amount,
                    "balance": round(balance, 2),
                }
            )
    return txs


def draw_statement(filename: str):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Margins
    margin_x = 20 * mm
    margin_y = 20 * mm

    # Random bank and customer info
    bank_name = random.choice(BANK_NAMES)
    customer_name = random_name()
    addr1, city, postcode = random_address()
    sort_code = random_sort_code()
    account_number = random_account_number()
    iban = random_iban_gb(sort_code, account_number)

    # Bank header (top-left)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin_x, height - margin_y, bank_name)

    c.setFont("Helvetica", 9)
    c.drawString(margin_x, height - margin_y - 12, "The Mound")
    c.drawString(margin_x, height - margin_y - 24, "Edinburgh EH1 1YZ")
    c.drawString(margin_x, height - margin_y - 36, "Telephone: 0345 721 3141")

    # Customer block (top-right)
    right_x = width - margin_x - 200
    y = height - margin_y
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x, y, customer_name)
    c.setFont("Helvetica", 9)
    c.drawString(right_x, y - 12, addr1)
    c.drawString(right_x, y - 24, city)
    c.drawString(right_x, y - 36, postcode)
    c.drawString(right_x, y - 48, "United Kingdom")

    # Account details
    y_details = y - 80
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y_details, "Account Summary")

    c.setFont("Helvetica", 9)
    y_details -= 14
    c.drawString(margin_x, y_details, f"Account Name: {customer_name}")
    y_details -= 12
    c.drawString(margin_x, y_details, f"Sort Code: {sort_code}")
    y_details -= 12
    c.drawString(margin_x, y_details, f"Account Number: {account_number}")
    y_details -= 12
    c.drawString(margin_x, y_details, f"IBAN: {iban}")

    # Statement period (random-ish)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    y_details -= 20
    c.setFont("Helvetica-Bold", 10)
    c.drawString(
        margin_x,
        y_details,
        f"Statement period: {start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}",
    )

    # Transactions header
    y_table = y_details - 30
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin_x, y_table, "Date")
    c.drawString(margin_x + 60, y_table, "Description")
    c.drawString(margin_x + 260, y_table, "Amount")
    c.drawString(margin_x + 330, y_table, "Balance")

    # Transactions
    txs = generate_transactions()
    y_table -= 14
    c.setFont("Helvetica", 8)

    for tx in txs:
        if y_table < margin_y + 40:
            # new page if needed
            c.showPage()
            y_table = height - margin_y - 40
            c.setFont("Helvetica-Bold", 9)
            c.drawString(margin_x, y_table, "Date")
            c.drawString(margin_x + 60, y_table, "Description")
            c.drawString(margin_x + 260, y_table, "Amount")
            c.drawString(margin_x + 330, y_table, "Balance")
            y_table -= 14
            c.setFont("Helvetica", 8)

        c.drawString(margin_x, y_table, tx["date"])
        c.drawString(margin_x + 60, y_table, tx["desc"][:35])
        c.drawRightString(
            margin_x + 320, y_table, f"{tx['amount']:,.2f}"
        )
        c.drawRightString(
            margin_x + 400, y_table, f"{tx['balance']:,.2f}"
        )
        y_table -= 12

    c.showPage()
    c.save()


def generate_many(output_dir="synthetic_statements", count=100):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, count + 1):
        filename = os.path.join(output_dir, f"statement_{i:03d}.pdf")
        print(f"Generating {filename}...")
        draw_statement(filename)


if __name__ == "__main__":
    generate_many(count=100)
