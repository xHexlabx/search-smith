**ลิงไต่ราว (Climbing Monkey)**

ลิงต๋อยต้องปีนเสา n ต้นที่สูง m เมตร เพื่อเก็บกล้วยทิพย์บนยอดเสา ระหว่างเสาที่อยู่ติดกันอาจมีกิ่งไม้เชื่อมในแนวนอน เมื่อลิงปีนไปพบกิ่งไม้ จะถูกบังคับให้ไต่ข้ามไปยังเสาอีกต้นเสมอ

ท่านเทพารักษ์ได้มอบกิ่งไม้วิเศษให้ลิงต๋อย 1 อัน ซึ่งสามารถใช้เชื่อมระหว่างเสาสองต้นที่อยู่ติดกัน ณ ระดับความสูงใดก็ได้ เพื่อให้สามารถเก็บกล้วยทิพย์ได้จำนวนมากที่สุด (อาจไม่จำเป็นต้องใช้ก็ได้)

**งานของคุณ**

จงเขียนโปรแกรมหาจำนวนกล้วยทิพย์ที่มากที่สุดที่ลิงต๋อยจะเก็บได้ เมื่อกำหนดเสาเริ่มต้นมาให้ และระบุว่ามีการใช้กิ่งไม้วิเศษหรือไม่

**ข้อมูลนำเข้า**

1.  **บรรทัดแรก** จำนวนเต็ม m, n, k (ความสูง, จำนวนเสา, จำนวนกิ่งไม้)
2.  **บรรทัดที่ 2** จำนวนกล้วยทิพย์บนยอดเสาแต่ละต้น
3.  **k บรรทัดถัดมา** ข้อมูลของกิ่งไม้แต่ละกิ่ง (หมายเลขเสาซ้าย, ระดับความสูง)
4.  **บรรทัดสุดท้าย** หมายเลขเสาที่ลิงต๋อยเริ่มปีน

**ข้อมูลส่งออก**

* **บรรทัดแรก** จำนวนกล้วยทิพย์ที่มากที่สุดที่เก็บได้
* **บรรทัดที่สอง** ระบุ "USE" หากใช้กิ่งไม้วิเศษ และ "NO" หากไม่ได้ใช้

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 20 5 6 | 9 |
| 7 5 3 9 4 | USE |
| 1 5 | |
| 1 6 | |
| 2 10 | |
| 1 12 | |
| 3 6 | |
| 3 13 | |
| 1 | |