**หาทำเลตั้งศูนย์บริการลูกค้า (location)**

ร้านคอมพิวเตอร์ต้องการเปิดศูนย์บริการลูกค้าในเมืองใหม่ซึ่งมีผังเมืองเป็นพื้นที่สี่เหลี่ยมย่อย $M \times N$ พื้นที่ และทราบจำนวนประชากรในแต่ละพื้นที่แล้ว ร้านจะเปิดศูนย์บริการเพียงร้านเดียวซึ่งจะครอบคลุมพื้นที่บริการขนาด $K \times K$

จงเขียนโปรแกรมที่มีประสิทธิภาพในการหาจำนวนประชากรรวมในทำเลพื้นที่บริการที่ดีที่สุด (มีประชากรรวมกันมากที่สุด)

**ข้อมูลเข้า**

1.  **บรรทัดแรก** เป็นเลขจำนวนเต็มบวกสองตัว M (แถว) และ N (หลัก) ($2\le M, N\le1,000$)
2.  **บรรทัดที่สอง** ระบุขนาดพื้นที่บริการ K ($0<K<M$ และ $0<K<N$)
3.  **บรรทัดที่สามถึง M+2** ระบุจำนวนประชากรในแต่ละพื้นที่ย่อย (ไม่เกิน 2,000 คนต่อพื้นที่)

**ข้อมูลส่งออก**

จำนวนประชากรภายในพื้นที่บริการที่ดีที่สุด

**ตัวอย่างที่ 1**

| ข้อมูลเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 5 10 | 31 |
| 2 | |
| 5 9 2 9 1 2 8 9 1 6 | |
| 9 1 3 9 8 4 2 1 5 7 | |
| 2 7 9 3 8 5 2 7 6 8 | |
| 1 6 2 1 7 7 1 9 4 1 | |
| 8 5 2 3 9 8 5 6 3 3 | |

**ตัวอย่างที่ 2**

| ข้อมูลเข้า | ข้อมูลส่งออก |
| :--- | :--- |
| 6 4 | 55 |
| 3 | |
| 8 7 5 1 | |
| 3 0 5 2 | |
| 3 3 2 9 | |
| 7 9 9 8 | |
| 3 4 5 9 | |
| 6 8 5 2 | |