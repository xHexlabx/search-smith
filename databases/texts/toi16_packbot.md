**หุ่นยนต์บรรจุสินค้า (PackBot)**

หุ่นยนต์ไดโนบอทมีหน้าที่บรรจุสินค้าตามรหัสคำสั่งและคำนวณราคาสุทธิ
* **สินค้า:** A-Z ทุกชนิดราคา 20 บาท
* **รูปแบบการบรรจุ (เชื่อมสินค้าหรือห่อสินค้า 2 ชิ้น):**
    1.  ใส่กล่อง (ลำดับความสำคัญต่ำสุด): ค่าบรรจุ 4% ของราคารวม
    2.  ใส่ถุงกระดาษ (ลำดับความสำคัญปานกลาง): ค่าบรรจุ 8% ของราคารวม
    3.  ใส่กล่องห่อของขวัญ (ลำดับความสำคัญสูงสุด): ค่าบรรจุ 16% ของราคารวม
* **การคำนวณ:** ค่าบรรจุให้ปัดเศษทศนิยมทิ้ง
* **ลำดับการทำงาน:** ทำตามลำดับความสำคัญของรูปแบบการบรรจุจากมากไปน้อย (3 -> 2 -> 1) ยกเว้นมีคำสั่งพิเศษในเครื่องหมาย `[]` ซึ่งต้องทำก่อน

**งานของคุณ**

เขียนโปรแกรมคำนวณยอดสุทธิของราคาที่ลูกค้าต้องจ่าย

**ข้อมูลนำเข้า**

เป็นสายอักขระรหัสคำสั่งการบรรจุสินค้า 1 บรรทัด (ยาวไม่เกิน 10,000 ตัวอักษร)

**ข้อมูลส่งออก**

ยอดสุทธิของราคาที่ลูกค้าต้องจ่าย

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| A3C1[F1G3H]3D | 153 |

**ตัวอย่างที่ 2**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| A3C1D2E | 92 |