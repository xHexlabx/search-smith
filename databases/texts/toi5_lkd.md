**ถอดรหัสแห่งความรัก (Love Key Decoder)**

โรมิโอส่งข้อความ (A-Z) ไปให้จูเลียตโดยเข้ารหัสแบบพิเศษ โรมิโอจะนำข้อความมาแปลงทีละตัวอักษรเป็นเลขฐานสองของรหัส ASCII (8 บิต) จากนั้นนำตัวเลขฐานสองที่ได้มาเข้ารหัสอีกครั้งด้วยเครื่องจักรเข้ารหัสซึ่งมี 4 สถานะ (S1, S2, S3, S4) และเริ่มต้นที่สถานะ S1

เครื่องจักรจะอ่านข้อมูลเข้า (p) ครั้งละ 1 บิต และเปลี่ยนสถานะไปตามเส้นเชื่อมพร้อมกับบันทึกข้อมูลออก (q) ขนาด 2 บิต ตามเงื่อนไข `p/q` ที่ระบุบนเส้นเชื่อม เมื่อเข้ารหัสครบทุกบิตแล้ว หากสถานะสุดท้ายไม่ใช่ S1 เครื่องจักรจะต้องทำงานเพิ่มเติม (โดยไม่มีข้อมูลเข้า) เพื่อกลับไปยังสถานะ S1 โดยเลือกเส้นทางที่เปลี่ยนสถานะน้อยที่สุด

**งานของคุณ**

จงเขียนโปรแกรมเพื่อช่วยจูเลียตถอดรหัสข้อความจากข้อมูลที่ได้รับ

**ข้อมูลนำเข้า**

* **บรรทัดแรก** ระบุจำนวนเต็ม N ($1\le N\le30$) แทนจำนวนบรรทัดของข้อมูลที่เข้ารหัสแล้ว
* **N บรรทัดถัดมา** แสดงข้อมูลที่เข้ารหัสแล้วครั้งละ 16 บิต (อาจน้อยกว่าสำหรับบรรทัดสุดท้าย)

**ข้อมูลส่งออก**

มีหนึ่งบรรทัดแสดงข้อความที่ถอดรหัสแล้ว

**ตัวอย่าง**

| ตัวอย่างที่ 1 | ตัวอย่างที่ 2 |
| :--- | :--- |
| **ข้อมูลนำเข้า** | **ข้อมูลนำเข้า** |
| 3 | 4 |
| 0011100010000110 | 0011101100000011 |
| 0100100010001000 | 1000101100001110 |
| 1011 | 1111101100001101 |
| | 0100101100111011 |
| **ข้อมูลส่งออก** | **ข้อมูลส่งออก** |
| WU | ABCD |