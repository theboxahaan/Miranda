# Sign-It
Deloitte-Hackathon
```python
import cv2
file_name = "grab.png"

src = cv2.imread(file_name, 1)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
b, g, r = cv2.split(src)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)
cv2.imwrite("test.png", dst)
```


```python
#pip install cloudant

from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey


#client = Cloudant.iam("<username>", "<apikey>")
client = Cloudant.iam("37915592-516e-4e57-bf99-5acb79819a25-bluemix", "feqaCgEnCYoV6_84XMXo8wwuyYpH_QWkWUZaKxKLByZP")
client.connect()

databaseName = "Credentials"

myDatabase = client.create_database(databaseName)
```
