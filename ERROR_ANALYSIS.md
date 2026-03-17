# 🔍 Error Analysis

This section analyzes failure cases to understand model limitations and improve robustness.

---

## 📌 Case 1
- **Text:** kinda calm now  
- **True:** overwhelmed  
- **Issue:** contradiction between text and label  
- **Reason:** noisy label or misleading wording  
- **Fix:** handle label noise, reduce confidence  

---

## 📌 Case 2
- **Text:** honestly got distracted again.  
- **True:** overwhelmed  
- **Issue:** ambiguous signal  
- **Reason:** "distracted" not strongly linked to overwhelmed  
- **Fix:** use metadata (stress, energy)  

---

## 📌 Case 3
- **Text:** i guess mind was all over the place.  
- **True:** neutral  
- **Issue:** conflicting interpretation  
- **Reason:** text suggests restless but labeled neutral  
- **Fix:** soft classification / label smoothing  

---

## 📌 Case 4
- **Text:** by the end it was fine...  
- **True:** neutral  
- **Issue:** vague input  
- **Reason:** weak emotional signal  
- **Fix:** use metadata + uncertainty flag  

---

## 📌 Case 5
- **Text:** at first still anxious a bit. after some time ...  
- **True:** neutral  
- **Issue:** emotion shift over time  
- **Reason:** model cannot capture temporal transitions  
- **Fix:** focus on final sentiment or sequence models  

---

## 📌 Case 6
- **Text:** during the session got distracted again. then ...  
- **True:** calm  
- **Issue:** mixed signals  
- **Reason:** both distraction and calm present  
- **Fix:** weighted interpretation  

---

## 📌 Case 7
- **Text:** at first kept thinking about work. then it shi...  
- **True:** restless  
- **Issue:** incomplete text  
- **Reason:** missing context  
- **Fix:** handle truncated inputs + uncertainty  

---

## 📌 Case 8
- **Text:** i noticed i felt distracted and i wasn't expec...  
- **True:** restless  
- **Issue:** weak + incomplete signal  
- **Fix:** improve text representation  

---

## 📌 Case 9
- **Text:** the cafe ambience helped me breathe slower and...  
- **True:** calm  
- **Issue:** subtle emotional expression  
- **Reason:** indirect calm signal  
- **Fix:** better semantic embeddings  

---

## 📌 Case 10
- **Text:** gradually my breathing slowed down before movi...  
- **True:** calm  
- **Issue:** implicit emotion  
- **Reason:** no direct emotional keywords  
- **Fix:** improve NLP feature extraction  

---

## 🧠 Key Insights

- Model struggles with **short and vague inputs**
- **Conflicting emotions** reduce prediction accuracy  
- **Incomplete or truncated text** causes errors  
- **Noisy labels** impact learning  
- Metadata improves **context understanding**  
- Confidence helps detect **uncertain predictions**

---

## ✅ Conclusion

Error analysis shows that ambiguity, noise, and incomplete inputs are major challenges.  
These are addressed using uncertainty modeling, metadata integration, and rule-based decision support.