**สารคดีออนไลน์ (NBK48)**

บริษัท NetNB ให้บริการรับชมสารคดีเรื่อง "ตามติดชีวิต NBK48" ซึ่งมี N ตอน โดยแต่ละตอนมีค่าบริการรับชม $p_i$ บาท (อาจเป็นค่าลบได้) ลูกค้าจะต้องรับชมเรียงตามลำดับตอนเสมอ โดยเริ่มจากตอนที่ 1, 2, 3, ...

**งานของคุณ**

จงเขียนโปรแกรมเพื่อหาจำนวนตอนของสารคดีที่มากที่สุดที่ลูกค้าแต่ละคนสามารถรับชมได้ภายใต้งบประมาณที่มีอยู่

**ข้อมูลนำเข้า**

1.  **บรรทัดที่ 1** จำนวนเต็ม N (จำนวนตอน, $1 \le N \le 100,000$) และ Q (จำนวนลูกค้า, $1 \le Q \le 100,000$)
2.  **บรรทัดที่ 2** จำนวนเต็ม N จำนวน ระบุค่ารับชม $p_i$ ของแต่ละตอน ($-10,000 \le p_i \le 10,000$)
3.  **Q บรรทัดต่อมา** แต่ละบรรทัดระบุจำนวนเงิน $q_j$ ของลูกค้าแต่ละคน

**ข้อมูลส่งออก**

มี Q บรรทัด แต่ละบรรทัดแสดงจำนวนตอนที่มากที่สุดที่ลูกค้าคนที่ j สามารถรับชมได้

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 5 2 | 2 |
| 10 20 15 30 60 | 4 |
| 44 | |
| 75 | |

**ตัวอย่างที่ 2**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 5 3 | 3 |
| 10 20 -10 30 60 | 4 |
| 31 | 0 |
| 52 | |
| 9 | |