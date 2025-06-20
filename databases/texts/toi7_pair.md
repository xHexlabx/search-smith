**คู่ตัวเลขเด่น (Pair)**

ให้ชุดของคู่อันดับจำนวนเต็มบวกมา n ชุด คือ $(a_1, b_1), (a_2, b_2), ..., (a_n, b_n)$ โดยที่ $a_i$ ทุกตัวไม่ซ้ำกัน และ $b_j$ ทุกตัวไม่ซ้ำกัน เราเรียกคู่อันดับ 2 คู่ $(a_i, b_i)$ และ $(a_j, b_j)$ ว่า **คู่ตัวเลขเด่น** ก็ต่อเมื่อ $a_i > a_j$ และ $b_i < b_j$

จงเขียนโปรแกรมที่มีประสิทธิภาพในการหาค่าผลรวมของ $a_i + a_j$ ทั้งหมดของทุกคู่ที่เป็นคู่ตัวเลขเด่น

**ข้อมูลนำเข้า**

1.  บรรทัดที่หนึ่งเป็นค่าของ n ($2 \le n \le 100,000$)
2.  บรรทัดที่สองเป็นค่าของคู่ตัวเลข $a_i$ และ $b_i$ จำนวน n คู่ โดยเรียงจากคู่ที่หนึ่งถึงคู่ที่ n (มีตัวเลขทั้งหมด 2n ตัว คั่นด้วยช่องว่าง)

**ข้อมูลส่งออก**

เป็นตัวเลขจำนวนเต็มบวกหนึ่งค่า ซึ่งแสดงถึงผลรวมของ $a_i + a_j$ ทั้งหมดของคู่ตัวเลขเด่น

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 6 | 78 |
| 2 1 7 6 9 3 18 4 3 5 | |

**ตัวอย่างที่ 2**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 4 | 39 |
| 1 4 3 2 2 3 7 1 | |