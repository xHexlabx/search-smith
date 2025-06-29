**ไวรัสโรคระบาด (Outbreak)**

ท่านได้รับมอบหมายให้จำลองการแพร่ของไวรัสสายพันธุ์ใหม่ในเมืองขอนแก่นสมาร์ทไดโนซิตี้เป็นเวลา T ชั่วโมง
**กฎการแพร่เชื้อ (เกิดขึ้นตามลำดับในทุก ๆ ชั่วโมง):**
1.  **ลดสุขภาพ:** คะแนนสุขภาพ H ของผู้ติดเชื้อทุกคนลดลง 1
2.  **กักตัว:** ผู้ติดเชื้อที่มี $H \le Q$ จะถูกนำไปกักตัวและรักษา (ออกจากพื้นที่จำลอง)
3.  **ติดเชื้อใหม่:** ผู้ปลอดเชื้อที่พิกัด (X,Y) จะติดเชื้อ ถ้า "ผลรวมของระดับการแพร่เชื้อ" จากเพื่อนบ้านที่ติดกัน (4 ทิศ) มีค่ามากกว่าหรือเท่ากับคะแนนสุขภาพ H ของตนเอง
4.  **ความสามารถในการแพร่เชื้อ:** ผู้ติดเชื้อจะสามารถแพร่เชื้อได้เมื่อ $H \le S$ โดยมี "ระดับการแพร่เชื้อ" เท่ากับ $F - H$ (F คือคะแนนสุขภาพสูงสุดที่เป็นไปได้)

**งานของคุณ**

จงเขียนโปรแกรมหาว่าเมื่อเวลาผ่านไป T ชั่วโมง จะมีผู้ติดเชื้อที่เหลือในพื้นที่รวมกี่คน และมีผู้ที่ถูกนำไปกักตัวและรักษากี่คน

**ข้อมูลนำเข้า**

1.  **บรรทัดที่ 1** N (จำนวนประชากร) และ T (เวลาที่จำลอง)
2.  **บรรทัดที่ 2** F, S, และ Q (คะแนนสุขภาพสูงสุด, เกณฑ์การแพร่เชื้อ, เกณฑ์การกักตัว)
3.  **N บรรทัดถัดไป** ข้อมูลประชากรแต่ละคน: X, Y, H, V (พิกัด, คะแนนสุขภาพ, สถานะการติดเชื้อ (1=ติด, 0=ไม่ติด))

**ข้อมูลส่งออก**

1.  **บรรทัดที่ 1** จำนวนผู้ติดเชื้อที่เหลืออยู่ในเมือง
2.  **บรรทัดที่ 2** จำนวนผู้ที่ถูกส่งไปกักตัวและรักษา

**ตัวอย่างที่ 1**

| ข้อมูลนำเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 5 5 | 1 |
| 100 60 10 | 2 |
| 0 0 61 1 | |
| 1 0 12 0 | |
| 2 0 10 0 | |
| 3 0 10 0 | |
| 4 0 30 0 | |