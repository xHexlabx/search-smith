**ป้อมภูผา (Peak)**

ในอาณาจักรสงขลา มีภูเขา n ลูกเรียงต่อกัน ภูเขาที่เหมาะสมสำหรับสร้างป้อมคือ "ภูเขาเด่น"
* **ภูเขาเด่น:** คือภูเขาที่สูงกว่าภูเขาที่อยู่ติดกันทั้งทางซ้ายและทางขวา (สำหรับภูเขาริมสุด ให้พิจารณาแค่เพื่อนบ้านด้านเดียว)

ทางการต้องการสร้างป้อมไม่เกิน k ป้อม โดยมีเงื่อนไขดังนี้:
1.  ต้องเลือกสร้างบนภูเขาเด่นที่มีความสูงมากที่สุดก่อน
2.  ไม่อนุญาตให้สร้างป้อมบนภูเขาเด่นที่มีระดับความสูงเท่ากันเกินหนึ่งป้อม

**งานของคุณ**

จงเขียนโปรแกรมหาความสูงของภูเขาเด่นที่เหมาะสมจะสร้างป้อม

**ข้อมูลนำเข้า**

1.  **บรรทัดที่ 1** จำนวนเต็ม n แทนจำนวนภูเขา ($5\le n\le5\times10^6$)
2.  **บรรทัดที่ 2** จำนวนเต็ม k แทนจำนวนป้อมสูงสุดที่สร้างได้ ($1\le k\le5\times10^5$)
3.  **n บรรทัดถัดมา** แต่ละบรรทัดคือความสูงของภูเขาแต่ละลูก

**ข้อมูลส่งออก**

* กรณีที่ไม่มีภูเขาเด่นเลย ให้แสดง `-1`
* มิฉะนั้น ให้แสดงความสูงของภูเขาเด่นที่ถูกเลือก โดยเรียงจากมากไปน้อย

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 10 | 90 |
| 2 | 45 |
| 40 | |
| 10 | |
| 90 | |
| 5 | |
| 45 | |
| 50 | |
| 65 | |
| 90 | |
| 35 | |
| 45 | |

**ตัวอย่างที่ 2**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 7 | -1 |
| 3 | |
| 3 | |
| 4 | |
| 6 | |
| 6 | |
| 6 | |
| 8 | |
| 9 | |