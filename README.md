# أداة تتبع معاملات العملات المشفرة (P2P Calculator)

<div dir="rtl">

## نظرة عامة

أداة متطورة لتتبع وتحليل معاملات العملات المشفرة والبنكية، مع التركيز على معاملات P2P و E-Voucher. تساعد هذه الأداة في حساب الأرباح وتتبع الأرصدة وتحليل التدفق النقدي بطريقة دقيقة وسهلة الاستخدام.

## الميزات الرئيسية

### تتبع المعاملات
- تتبع معاملات P2P (شراء وبيع USDT مقابل عملات مختلفة)
- تتبع معاملات E-Voucher (استلام درهم من العمال وبيع USDT مقابل جنيه مصري)
- دعم لعملات متعددة (AED، EGP، USDT، وإمكانية إضافة عملات أخرى)

### حساب الأرباح
- حساب أرباح P2P باستخدام طريقة FIFO (الوارد أولاً يصرف أولاً)
- حساب أرباح E-Voucher بشكل منفصل (الدرهم المستلم - تكلفة USDT المباع)
- عرض تفاصيل دقيقة لكيفية حساب الأرباح

### تحليلات متقدمة
- تدفق نقدي مفصل لكل عملية
- تحليل تطور الأرصدة مع الوقت
- رسوم بيانية توضيحية للأرصدة والأرباح
- تحليلات منفصلة لمعاملات P2P و E-Voucher

### واجهة مستخدم متطورة
- تصميم عصري وسهل الاستخدام
- عرض المبالغ بدقة عالية
- إمكانية تخصيص الإعدادات
- دعم كامل للغة العربية

## متطلبات التشغيل

- Python 3.7+
- المكتبات: streamlit, pandas, plotly, openpyxl, numpy

## طريقة التثبيت

1. استنساخ المستودع:
```
git clone https://github.com/SpyMeOrg/p2p-calculator.git
cd p2p-calculator
```

2. تثبيت المكتبات المطلوبة:
```
pip install -r requirements.txt
```

3. تشغيل التطبيق:
```
streamlit run p2p_tracker.py
```

## طريقة الاستخدام

### الإعدادات الأولية
1. أدخل الأرصدة الافتتاحية لكل عملة (AED، EGP، USDT)
2. أدخل سعر USDT الافتتاحي (السعر الذي تم شراء USDT الافتتاحي به)
3. أدخل إجمالي مبلغ الدرهم المستلم من العمال لمعاملات E-Voucher (اختياري)

### استيراد البيانات
1. قم برفع ملف Excel يحتوي على معاملات P2P و E-Voucher
2. يمكن استخدام ملف `sample_data_new.xlsx` كنموذج

### استعراض النتائج
- **لوحة التحكم**: عرض ملخص للأرصدة والأرباح
- **التدفق النقدي**: عرض تدفق نقدي مفصل لكل عملية
- **المعاملات**: عرض تفاصيل جميع المعاملات
- **تحليل P2P**: تحليلات خاصة بمعاملات P2P
- **تحليل E-Voucher**: تحليلات خاصة بمعاملات E-Voucher
- **التصحيح**: عرض تفاصيل حساب الربح لكل عملية بيع

## هيكل ملف الإكسل المطلوب

يجب أن يحتوي ملف الإكسل على الأعمدة التالية:

| العمود | الوصف | مثال |
|--------|-------|-------|
| Reference | مرجع المعاملة (اختياري) | TX0001 |
| Type | نوع المعاملة | Buy/Sell |
| Currency | العملة المستخدمة | AED/EGP |
| Amount | المبلغ قبل الرسوم (اختياري) | 1000.00 |
| Real Amount | المبلغ الفعلي بعد الرسوم | 990.00 |
| AED/EGP | سعر صرف الدرهم مقابل الجنيه | 13.75 |
| Usdt B | كمية USDT قبل الرسوم (اختياري) | 270.27 |
| USDT | كمية USDT النهائية | 267.57 |
| Price | سعر الوحدة | 3.70 |
| Fees | الرسوم (اختياري) | 10.00 |
| Status | حالة المعاملة | COMPLETED |
| Date | تاريخ ووقت المعاملة | 01/01/2023, 10:00:00 |
| TradeType | نوع التداول | P2P/E-Voucher |

## ملاحظات مهمة

- يتم حساب أرباح P2P فقط من معاملات P2P
- يتم حساب أرباح E-Voucher بشكل منفصل
- الجنيه المصري (EGP) المرسل للعائلات لا يحتسب كأصل
- القيمة الإجمالية تحسب فقط من الدرهم الإماراتي (AED) وقيمة USDT

## المساهمة في المشروع

نرحب بمساهماتكم لتحسين هذا المشروع. يمكنكم:
1. عمل Fork للمستودع
2. إنشاء فرع جديد للميزة التي ترغبون في إضافتها
3. إرسال Pull Request مع وصف مفصل للتغييرات

## الترخيص

هذا المشروع مرخص تحت [MIT License](LICENSE).

</div>

---

# Crypto-bank Transaction Tracking Tool (P2P Calculator)

<div dir="ltr">

## Overview

An advanced tool for tracking and analyzing cryptocurrency and bank transactions, focusing on P2P and E-Voucher transactions. This tool helps calculate profits, track balances, and analyze cash flow in an accurate and user-friendly way.

## Key Features

### Transaction Tracking
- Track P2P transactions (buying and selling USDT for various currencies)
- Track E-Voucher transactions (receiving AED from workers and selling USDT for EGP)
- Support for multiple currencies (AED, EGP, USDT, with the ability to add more)

### Profit Calculation
- Calculate P2P profits using the FIFO method (First In, First Out)
- Calculate E-Voucher profits separately (AED received - cost of USDT sold)
- Display detailed breakdown of profit calculations

### Advanced Analytics
- Detailed cash flow for each transaction
- Analysis of balance evolution over time
- Illustrative charts for balances and profits
- Separate analytics for P2P and E-Voucher transactions

### Advanced User Interface
- Modern and user-friendly design
- High-precision display of amounts
- Customizable settings
- Full Arabic language support

## Requirements

- Python 3.7+
- Libraries: streamlit, pandas, plotly, openpyxl, numpy

## Installation

1. Clone the repository:
```
git clone https://github.com/SpyMeOrg/p2p-calculator.git
cd p2p-calculator
```

2. Install required libraries:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run p2p_tracker.py
```

## Usage

### Initial Settings
1. Enter initial balances for each currency (AED, EGP, USDT)
2. Enter the initial USDT rate (the rate at which the initial USDT was purchased)
3. Enter the total AED amount received from workers for E-Voucher transactions (optional)

### Import Data
1. Upload an Excel file containing P2P and E-Voucher transactions
2. You can use `sample_data_new.xlsx` as a template

### View Results
- **Dashboard**: View a summary of balances and profits
- **Cash Flow**: View detailed cash flow for each transaction
- **Transactions**: View details of all transactions
- **P2P Analysis**: View analytics specific to P2P transactions
- **E-Voucher Analysis**: View analytics specific to E-Voucher transactions
- **Debug**: View detailed profit calculation for each sell transaction

## Excel File Structure

The Excel file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Reference | Transaction reference (optional) | TX0001 |
| Type | Transaction type | Buy/Sell |
| Currency | Currency used | AED/EGP |
| Amount | Amount before fees (optional) | 1000.00 |
| Real Amount | Actual amount after fees | 990.00 |
| AED/EGP | AED to EGP exchange rate | 13.75 |
| Usdt B | USDT amount before fees (optional) | 270.27 |
| USDT | Final USDT amount | 267.57 |
| Price | Unit price | 3.70 |
| Fees | Fees (optional) | 10.00 |
| Status | Transaction status | COMPLETED |
| Date | Transaction date and time | 01/01/2023, 10:00:00 |
| TradeType | Trade type | P2P/E-Voucher |

## Important Notes

- P2P profits are calculated only from P2P transactions
- E-Voucher profits are calculated separately
- EGP sent to families is not counted as an asset
- Total value is calculated only from AED and USDT value

## Contributing

We welcome contributions to improve this project. You can:
1. Fork the repository
2. Create a new branch for the feature you want to add
3. Submit a Pull Request with a detailed description of the changes

## License

This project is licensed under the [MIT License](LICENSE).

</div>
