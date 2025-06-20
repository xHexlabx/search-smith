**ท่อระบายน้ำ (Sewer)**

เมืองแห่งหนึ่งมีพื้นที่เป็นรูปสี่เหลี่ยมขนาด a แถวคูณ b คอลัมน์ แต่ละเขตจะมีท่อระบายน้ำเชื่อมต่อกับเขตเพื่อนบ้านหรือไม่ก็ได้ น้ำจะเริ่มปล่อยจากเขต (1, 1) และสามารถไหลได้ 4 ทิศทาง (เหนือ, ใต้, ตะวันออก, ตะวันตก) โดยใช้เวลา 1 หน่วยในการเดินทางจากเขตหนึ่งไปอีกเขตหนึ่ง และน้ำไม่สามารถไหลย้อนกลับได้

รูปแบบท่อระบายน้ำในแต่ละเขตจะถูกกำหนดโดยพิจารณาการเชื่อมต่อไปยังทิศตะวันออกและทิศใต้เท่านั้น:
* **R**: เชื่อมกับเขตทิศตะวันออก
* **D**: เชื่อมกับเขตทิศใต้
* **B**: เชื่อมกับทั้งเขตทิศตะวันออกและทิศใต้
* **N**: ไม่เชื่อมกับทั้งสองทิศ

จงเขียนโปรแกรมเพื่อคำนวณหาระยะเวลาที่น้อยที่สุดที่น้ำทิ้งอย่างน้อย 2 สายจะมาบรรจบกัน พร้อมทั้งบอกพิกัดของเขตนั้น (รับประกันว่ามีเขตที่น้ำสองสายมาบรรจบกันเร็วที่สุดเพียงเขตเดียวเสมอ)

**ข้อมูลนำเข้า**

1.  บรรทัดแรกเป็นค่าของตัวแปร a และ b ($2 \le a, b \le 100$)
2.  บรรทัดที่สองถึง a+1 แต่ละบรรทัดมีตัวอักษร b ตัว คั่นด้วยช่องว่าง ระบุสถานะของท่อระบายน้ำ

**ข้อมูลส่งออก**

1.  บรรทัดแรกเป็นจำนวนเต็ม แสดงถึงช่วงเวลาที่น้ำทิ้งมาบรรจบกัน
2.  บรรทัดที่สองเป็นจำนวนเต็ม 2 ตัว คั่นด้วยช่องว่าง ซึ่งเป็นพิกัด (แถว, คอลัมน์) ที่น้ำทิ้งมาบรรจบกัน

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 4 4 | 5 |
| B R D N | 3 3 |
| D R B D | |
| R R R D | |
| N N N N | |

**ตัวอย่างที่ 2**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 3 4 | 5 |
| B B B D | 2 4 |
| D N R B | |
| R R R N | |