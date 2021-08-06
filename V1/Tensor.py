import numpy as np

class Tensor (object):
      """ 
      creators -> список Тензоров для создания текущего тензора z = x + y , creators = [x, y]
      creation_op -> Содержит атрибут для созданяи этого Тензора , z = x + y, creation_op = Add
      children подсчитывающий количество градиентов, полученных от каждого потомка в процессе обратного распространения.
      """
      def __init__(self,data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        
        self.data = np.array(data)
        self.autograd = autograd
        self.grad = None
        if (id is None):
            self.id = np.random.randint(0,100000)
        else:
            self.id = id
        
        self.creators = creators
        self.creation_op = creation_op
        self.children = {}
        
        # Скорректировать число потомков
         # данного тензора 
        # Здесь мы инициализирцем в создании чьих векторов мы учасвтвуем и сколько раз
        if (creators is not None):
            for c in creators:
                if(self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
      #Проверить, получил ли тензор
      #градиенты от всех потомков
      def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            if (cnt != 0):
                return False
        return True 
    
    # Grad сам градиент grad_origin -> от кого идет
      def backward(self, grad=None, grad_origin=None):
        if (self.autograd):
 
            if (grad is None):
                grad = Tensor(np.ones_like(self.data))

            

            if (grad_origin is not None):
                if (self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                     #Уменьшаем сколько раз участвовали
                    self.children[grad_origin.id] -= 1

            if (self.grad is None):
              # Если градиент пусто
                self.grad = grad
            else:
                self.grad += grad
            
            # grads must not have grads of their own
            assert grad.autograd == False
            
            # only continue backpropping if there's something to
            # backprop into and if all gradients (from children)
            # are accounted for override waiting for children if
            # "backprop" was called on this variable directly
            #  Если есть создатели тензора и получили градиенты от всех потомков и нет от кого идет градиент
            if (self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None)):

                if (self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                if (self.creation_op == "sub"):
                    self.creators[0].backward(Tensor(self.grad.data), self)
                    self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

                if (self.creation_op == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)                    
                    
                if (self.creation_op == "mm"):
                    c0 = self.creators[0]
                    c1 = self.creators[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)
                    
                if (self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())

                if ("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.expand(dim,
                                                               self.creators[0].data.shape[dim]))

                if ("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))
                    
                if (self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())
                
                if (self.creation_op == "sigmoid"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self * (ones - self)))
                
                if (self.creation_op == "tanh"):
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (ones - (self * self)))

                if (self.creation_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx))    
    
    # +                
      def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="add")
        return Tensor(self.data + other.data)
    
    # - перед классом
      def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * -1)
    
    # -
      def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="sub")
        return Tensor(self.data - other.data)
    
    # "*"
      def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self,other],
                          creation_op="mul")
        return Tensor(self.data * other.data)    


      """
      Функция sum выполняет сложение элементов тензора по измерениям
      х = Tensor(np.array([[1,2,3],
                      [4,5,6]]))

      х. sum(0) вернет матрицу 1 3 (вектор с тремя элементами), а х. sum(1) вернет
      матрицу 2x1 (вектор с двумя элементами):

      x.sum(0) ------► аггау([5, 7, 9]) x.sum(l) ------► array([ 6, 15])

      """
      def sum(self, dim):
        if (self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
      """
      Функция expand используется для обратного распространения операции . sum().
      Она копирует данные по измерению. Для той же матрицы х копирование по первому измерению даст две копии тензора
      То есть если .sum() удаляет размерность (матрицу 2x3 превращает в вектор
      длиной 2 или 3), то expand добавляет измерение
      """
      def expand(self, dim,copies):

        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if (self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)
    
     #  Трансопнирование
      def transpose(self):
        if (self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        
        return Tensor(self.data.transpose())
    
    # Скалярное умножение (dot)
      def mm(self, x):
        if (self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self,x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

      def sigmoid(self):
        if (self.autograd):
          return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))

      def tanh(self):
        if (self.autograd):
          return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

      def cross_entropy(self, target_indices):
      

        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t),-1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()
    
        if (self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out

        return Tensor(loss)


      def __repr__(self):
        return str(self.data.__repr__())
    
      def __str__(self):
        return str(self.data.__str__())


# Класс для изменения весов
# Стохастический градиентный спуск
class SGD(object):
    
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha
    
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):

        for p in self.parameters:
            
            p.data -= p.grad.data * self.alpha
            if (zero):
                p.grad.data *= 0





