**จอมกดส่งข้อความ (SMS Thumb)**

กำหนดปุ่มกดโทรศัพท์มือถือและการวนของตัวอักษรเมื่อกดปุ่มซ้ำๆ ปุ่ม 1 ใช้สำหรับลบ (DEL) การเลื่อนนิ้วไปยังปุ่มใหม่ (หรือปุ่มเดิม) จะนับการกดเริ่มจากตัวอักษรแรกของปุ่มนั้นเสมอ

จงเขียนโปรแกรมหาข้อความที่พิมพ์จากข้อมูลการสังเกตการณ์

**ข้อมูลนำเข้า**

  * **บรรทัดแรก** จำนวนครั้งที่เลือกปุ่มกด N ($1\\le N\\le80$)
  * **บรรทัดที่สอง** ปุ่มแรกที่กด S ($1\\le S\\le9$) และจำนวนครั้งที่กด M ($1\\le M\\le4096$)
  * **N - 1 บรรทัดถัดมา** แต่ละบรรทัดประกอบด้วยตัวเลข 3 จำนวน: ทิศทางแนวนอน H, ทิศทางแนวตั้ง V, และจำนวนครั้งที่กด M
      * H: -2 ถึง 2 (ลบคือซ้าย, บวกคือขวา)
      * V: -2 ถึง 2 (ลบคือบน, บวกคือล่าง)

**ข้อมูลส่งออก**

แสดงข้อความที่พิมพ์ในบรรทัดเดียว ถ้าไม่ได้พิมพ์อะไรเลยให้แสดงคำว่า `null`

**ตัวอย่าง**

| ตัวอย่างที่ 1 | ตัวอย่างที่ 2 | ตัวอย่างที่ 3 |
| :--- | :--- | :--- |
| **ข้อมูลนำเข้า** | **ข้อมูลนำเข้า** | **ข้อมูลนำเข้า** |
| 4 | 2 | 5 |
| 5 3 | 9 6 | 3 3 |
| 1 0 3 | -2 -2 5 | 0 0 2 |
| -1 1 3 | | -2 0 1 |
| 1 -2 2 | | 2 1 3 |
| | | 0 1 2 |
| **ข้อมูลส่งออก** | **ข้อมูลส่งออก** | **ข้อมูลส่งออก** |
| LOVE | null | FOX |